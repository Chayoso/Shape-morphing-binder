"""Read-only accounting of stored momentum and hybrid position updates on CUDA/CPU."""
from __future__ import annotations

import torch
import warp as wp

NAMES = ('mpm_advection', 'bond_residual', 'surface_u', 'layer_residual')


@torch.no_grad()
def collect_rollout(tr, dt):
    """Read final validated buffers; never replay or mutate a trajectory."""
    initial = wp.to_torch(tr.x[0])
    zero = torch.zeros_like(initial)
    u_step = zero
    if tr.layer:
        active = (wp.to_torch(tr.layer_mask) >= .5) & (wp.to_torch(tr.pin) <= .5)
        u_step = (float(tr.layer_frac_u) * wp.to_torch(tr.layer_ug)
                  * wp.to_torch(tr.layer_u) * active)[:, None] * wp.to_torch(tr.layer_nrm)
    sums = torch.zeros((len(NAMES), *initial.shape), device=initial.device)
    squares, dots = [], []
    closure = initial.new_zeros(())
    for t in range(tr.T):
        before = wp.to_torch(tr.x[t])
        after = wp.to_torch(tr.x[t+1])
        prelayer = wp.to_torch(tr.xu[t+1]) if tr.layer else after
        velocity = wp.to_torch(tr.v[t+1])
        total = after-before
        components = torch.stack((dt*velocity, prelayer-before-dt*velocity,
                                  u_step, after-prelayer-u_step))
        sums += components
        squares.append(torch.cat((components.square().sum(-1), total.square().sum(-1)[None])))
        dots.append((components*total[None]).sum(-1))
        closure = torch.maximum(closure, (components.sum(0)-total).norm(dim=-1).max())
    return dict(squares=torch.stack(squares), dots=torch.stack(dots), sums=sums,
                closure_wu=float(closure), terminal_mpm=velocity.clone(),
                terminal_geometry=(total/dt).clone(), dt=float(dt))


@torch.no_grad()
def summarize(accounting, start, rollout_end, before_pic, after_pic, promoted,
              start_arrived, end_arrived, pinned):
    """Cohorts use the frozen plan and actual promoted endpoint, after commit operators.

    RMS component magnitudes are not additive. Signed fractions expose cancellation;
    their sum is one except for roundoff or a zero reference displacement.
    """
    free = ~pinned.bool()
    arrived = start_arrived.bool() & end_arrived.bool()
    cohorts = dict(all=torch.ones_like(free), arrived_free=free & arrived,
                   transit_free=free & ~arrived, pinned_at_start=~free)
    net = promoted-start
    commits = torch.stack((before_pic-rollout_end, after_pic-before_pic, promoted-after_pic))
    terminal_promoted = accounting['terminal_geometry'] + (promoted-rollout_end)/accounting['dt']
    window = torch.cat((accounting['sums'], commits))
    names = NAMES + ('commit_other', 'commit_pic', 'commit_shift')
    result = dict(dt=accounting['dt'], T=len(accounting['squares']),
                  substep_closure_max_wu=accounting['closure_wu'],
                  window_closure_max_wu=float((window.sum(0)-net).norm(dim=-1).max()),
                  cohort_definition='window-start pins; arrived_free is arrival at BOTH start and promoted end against the frozen full plan',
                  velocity_definition='geometry is last rollout dx/dt before commit; promoted_geometry includes all endpoint position corrections',
                  residual_definition='bond_residual includes k_update arithmetic roundoff; layer_residual includes relaxation and layer arithmetic roundoff',
                  cohorts={})
    for name, mask in cohorts.items():
        count = int(mask.sum())
        if not count:
            result['cohorts'][name] = dict(particles=0)
            continue
        sq = accounting['squares'][:, :, mask]
        dot = accounting['dots'][:, :, mask]
        step_den = sq[:, -1].sum()
        window_den = net[mask].square().sum()
        vm = accounting['terminal_mpm'][mask]
        vg = accounting['terminal_geometry'][mask]
        vp = terminal_promoted[mask]
        result['cohorts'][name] = dict(
            particles=count,
            step_rms_wu={key: sq[:, i].mean(-1).sqrt().tolist() for i, key in enumerate(NAMES+('actual',))},
            step_signed_fraction={key: float(dot[:, i].sum()/step_den) if step_den > 0 else None for i, key in enumerate(NAMES)},
            terminal_speed_wu_s={
                'mpm_mean': float(vm.norm(dim=-1).mean()), 'geometry_mean': float(vg.norm(dim=-1).mean()),
                'mpm_p95': float(torch.quantile(vm.norm(dim=-1), .95)),
                'geometry_p95': float(torch.quantile(vg.norm(dim=-1), .95)),
                'difference_rms': float((vg-vm).square().sum(-1).mean().sqrt()),
                'promoted_geometry_mean': float(vp.norm(dim=-1).mean()),
                'promoted_geometry_p95': float(torch.quantile(vp.norm(dim=-1), .95)),
                'promoted_difference_rms': float((vp-vm).square().sum(-1).mean().sqrt())},
            window_component_rms_wu={key: float(window[i, mask].square().sum(-1).mean().sqrt()) for i, key in enumerate(names)},
            window_signed_fraction={key: float((window[i, mask]*net[mask]).sum()/window_den) if window_den > 0 else None for i, key in enumerate(names)},
            window_net_rms_wu=float(net[mask].square().sum(-1).mean().sqrt()))
    return result
