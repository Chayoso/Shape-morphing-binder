"""Raw-state coverage/motion audit. No renderer or optimization-loss operator is used.

Density is a neighborhood-count diagnostic, not a proof of topology or watertightness.
Use matching window indices and discretisation for comparisons, not normalized time
between runs of different length. Uncompressed NPZ frames are memory mapped.
"""
import argparse
import json
from pathlib import Path
import struct
import zipfile
import numpy as np
from scipy.spatial import cKDTree

ARM = 'render_full_dt_iso_nn'


def frames_array(path):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('frames.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            with np.load(path) as z:
                return z['frames']
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            npy_header = stream.tell()
        with open(path, 'rb') as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        offset = info.header_offset + 30 + name_len + extra_len + npy_header
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape,
                     order='F' if fortran else 'C')


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values))) if len(values) else None


def audit(prefix, every_frame=False):
    meta = json.loads(Path(str(prefix) + '.json').read_text())
    arm = meta['arms'][ARM]; cfg = arm['config']
    records = [r for r in arm['history'] if r.get('frame_end') and not r.get('null_commit')]
    path = str(prefix) + '_' + ARM + '.npz'
    frames = frames_array(path)
    with np.load(path) as z:
        target, source = z['tgt'], z['src']
        pins, pin_at = z['pinned'], z['pinned_at']
        delivered = min(int(z['deliver_n']), len(frames))
    spacing = float(np.median(cKDTree(source).query(source, k=2, workers=8)[0][:, 1]))
    rcov = float(np.median(cKDTree(target).query(target, k=9, workers=8)[0][:, 8]))
    sample = np.linspace(0, len(source)-1, min(20000, len(source)), dtype=np.int64)
    frame_records = {int(r['frame_end'])-1: r for r in records}
    committed = [r for r in records if int(r['frame_end']) <= delivered]
    simulated_end = max((int(r['frame_end']) for r in committed), default=0)
    ids = range(delivered) if every_frame else sorted({0} | {i for i in frame_records if i < delivered})
    density = []
    for i in ids:
        x = np.asarray(frames[i]); tree = cKDTree(x)
        counts = np.asarray(tree.query_ball_point(x[sample], rcov, return_length=True, workers=8))-1
        top = x[:, 1] > 2.3
        tc = np.asarray(tree.query_ball_point(x[top], rcov, return_length=True, workers=8))-1 if top.any() else np.array([])
        rec = frame_records.get(i, {})
        density.append(dict(frame=int(i), window=(int(rec['animation'])+1 if rec else None),
                            sampled_under_half=float((counts < 4).mean()), top_n=int(top.sum()),
                            top_density=(float(tc.mean()/8) if len(tc) else None),
                            top_under_half=(float((tc < 4).mean()) if len(tc) else None)))
    # Read admitted particles at their actual commit boundary; final membership alone
    # cannot establish a historical constraint when release modes are enabled.
    pin_motion = None
    end_pins = np.zeros(len(pins), bool)
    release = any(cfg.get(k) for k in ('settle_pin_yield', 'settle_pin_follow', 'settle_pin_kkt'))
    if not release:
        worst = 0.; checked = 0; particle_maxima = []
        ani_to_frame = {int(r['animation'])+1: int(r['frame_end'])-1 for r in records}
        for when, frame in ani_to_frame.items():
            if frame < delivered:
                end_pins |= pins & (pin_at == when)
        for when in np.unique(pin_at[pins]):
            if int(when) not in ani_to_frame:
                continue
            start = ani_to_frame[int(when)]; selected = pins & (pin_at == when)
            if start >= delivered-1:
                continue
            anchor = np.asarray(frames[start])[selected].copy()
            maxima = np.zeros(len(anchor), np.float32)
            for i in range(start+1, delivered):
                maxima = np.maximum(maxima, np.linalg.norm(np.asarray(frames[i])[selected]-anchor, axis=1))
            particle_maxima.append(maxima)
            worst = max(worst, float(maxima.max()))
            checked += int(selected.sum())
        maxima = np.concatenate(particle_maxima) if particle_maxima else np.array([])
        pin_motion = dict(checked_particles=checked, max_wu=worst, max_sp=worst/spacing,
                          moved_particles_exact=int((maxima > 0).sum()),
                          per_particle_max_drift_sp=stats(maxima/spacing))
    final = np.asarray(frames[delivered-1]); tree = cKDTree(final)
    count = np.asarray(tree.query_ball_point(final, 2*spacing, return_length=True, workers=8))
    surface = count < 0.6*np.median(count)
    chosen = np.flatnonzero(surface & ~end_pins)
    if len(chosen) > 20000:
        chosen = chosen[np.linspace(0, len(chosen)-1, 20000, dtype=np.int64)]
    tail = None
    if not release and len(chosen) and simulated_end > 3:
        _, neighbors = tree.query(final[chosen], k=33, workers=8)
        normal = final[chosen]-final[neighbors].mean(1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        moves = np.diff(np.asarray(frames[max(0, int(simulated_end*.9)-1):simulated_end, chosen]), axis=0)
        signed = (moves*normal).sum(2)
        tangent = np.linalg.norm(moves-signed[:, :, None]*normal, axis=2)
        moving = (np.linalg.norm(moves[1:], axis=2) > 1e-4*spacing) & (np.linalg.norm(moves[:-1], axis=2) > 1e-4*spacing)
        reversal = (moves[1:]*moves[:-1]).sum(2) < 0
        tail = dict(cohort='surface and unpinned at delivered endpoint; fixed cohort across tail',
                    particles=len(chosen), normal_sp_per_archived_step=stats(np.abs(signed).ravel()/spacing),
                    tangential_sp_per_archived_step=stats(tangent.ravel()/spacing),
                    reversal_fraction=float(reversal[moving].mean()) if moving.any() else 0.)
    return dict(prefix=str(prefix), n=len(source), T=cfg['T'], animations=cfg['animations'],
                discretisation=meta.get('provenance', {}).get('mpm'), loss_res=cfg.get('loss_res'),
                code_hash=meta.get('provenance', {}).get('code_hash'),
                stop_after_windows=cfg.get('stop_after_windows', 0), native_nn_spacing=spacing,
                target_r8=rcov, delivered_frames=delivered, metrics=arm['metrics'], guards=arm['guards'],
                simulated_frames=simulated_end, terminal_v_mean=(committed[-1].get('v_mean') if committed else None),
                active_pins_at_delivered_end=(int(end_pins.sum()) if not release else None),
                body_ctrl=cfg.get('body_ctrl', False), density=density, pin_motion=pin_motion,
                tail_unpinned_surface=tail, windows=[{k:r.get(k) for k in ('animation','body_rms_wu','body_nodes',
                   'active_pin_motion_max','pinned_frac','arrived_end_frac','Jmin_traj','move','pace_lead_applied')} for r in records])


def main():
    p = argparse.ArgumentParser(); p.add_argument('prefix', type=Path)
    p.add_argument('--out', required=True, type=Path); p.add_argument('--every-frame', action='store_true')
    args = p.parse_args(); result = audit(args.prefix, args.every_frame)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ('prefix','n','body_ctrl','pin_motion','tail_unpinned_surface','density')}))


if __name__ == '__main__':
    main()
