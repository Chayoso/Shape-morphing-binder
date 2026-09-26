"""Keep an actively pinned material point's splat attributes fixed after admission."""
from __future__ import annotations
import numpy as np
import torch


def pin_start_frames(pinned, admitted_at, history, config):
    if any(config.get(k) for k in ('settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt')):
        raise ValueError('appearance locking requires monotone pins; release modes need per-frame active masks')
    starts = np.full(len(pinned), np.iinfo(np.int64).max, np.int64)
    commits = {int(r['animation'])+1: int(r['frame_end'])-1 for r in history
               if r.get('frame_end') and not r.get('null_commit')}
    for admission in np.unique(np.asarray(admitted_at)[pinned]):
        if int(admission) not in commits:
            raise ValueError(f'pin admission {admission} has no accepted commit boundary')
        starts[np.asarray(pinned) & (np.asarray(admitted_at) == admission)] = commits[int(admission)]
    return starts


class SettledAppearance:
    def __init__(self, starts, device):
        self.starts = torch.as_tensor(starts, device=device)
        self.anchored = torch.zeros(len(starts), dtype=torch.bool, device=device)
        self.values = None

    def apply(self, frame, x, normals, sigma, support):
        current = (x, normals, sigma)
        if self.values is None:
            self.values = [torch.zeros_like(v) for v in current]
        # Do not hide a physical pin violation by freezing only its displayed position.
        if self.anchored.any() and not torch.equal(x[self.anchored], self.values[0][self.anchored]):
            raise ValueError('an active pin moved; repair physical state before locking appearance')
        newly = (frame >= self.starts) & ~self.anchored
        for stored, value in zip(self.values, current):
            stored[newly] = value[newly]
        self.anchored |= newly
        result = []
        for value, stored in zip(current[1:], self.values[1:]):
            out = value.clone(); out[self.anchored] = stored[self.anchored]; result.append(out)
        # Density support remains live: locking it could conceal departing neighbors.
        return (*result, support)


def validate_pinned_frames(frames, starts, stop):
    """Check every raw delivered frame, including admission-to-first-render intervals."""
    anchors = np.zeros_like(frames[0])
    for start in np.unique(starts[starts < stop]):
        selected = starts == start
        anchors[selected] = frames[int(start)][selected]
    for i in range(stop):
        active = starts < i
        if active.any() and not np.array_equal(frames[i][active], anchors[active]):
            raise ValueError(f'active pins moved in raw frame {i}; appearance lock refused')


def validate_pinned_frames_cuda(frames, starts, stop, device='cuda'):
    """GPU check of every raw frame, including frames omitted by export stride."""
    if torch.device(device).type != 'cuda':
        raise ValueError('CUDA pin validation requires a CUDA device')
    starts = torch.as_tensor(starts, device=device)
    anchors = torch.zeros((len(starts), 3), device=device)
    for index in range(stop):
        x = torch.as_tensor(np.asarray(frames[index], np.float32), device=device)
        new = starts == index
        anchors[new] = x[new]
        active = starts < index
        if not torch.equal(x[active], anchors[active]):
            raise ValueError(f'active pins moved at raw frame {index}; rendering refused')
