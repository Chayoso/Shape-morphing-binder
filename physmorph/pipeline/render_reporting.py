"""Accepted-step render influence, with explicit non-causal interpretations."""
import json
import math
from pathlib import Path
import statistics

import torch


@torch.no_grad()
def accepted_render_step(physics, render, leaves, previous, names, weight,
                         loss_before, loss_after, parts_before, parts_after,
                         endpoint_before, endpoint_after, iteration, backtracks, alpha):
    channels = {}
    pn2, rn2 = 0., 0.
    available = physics is not None and render is not None
    for i, (p, old, name) in enumerate(zip(leaves, previous, names)):
        delta = p.detach()-old
        if not available:
            channels[name] = dict(accepted_control_delta_norm=float(delta.norm()),
                                  direction_statistics=None)
            continue
        pg, rg = physics[i], render[i]
        pn, rn = float(pg.norm()), float(rg.norm())
        pn2 += pn*pn; rn2 += rn*rn
        channels[name] = dict(physics_direction_norm=pn, render_direction_norm=rn,
            nominal_render_share=weight*rn/max(pn+weight*rn, 1e-30),
            accepted_control_delta_norm=float(delta.norm()),
            optimizer_render_direction_dot_delta=float((rg*delta).sum()),
            weighted_optimizer_render_direction_dot_delta=weight*float((rg*delta).sum()),
            physics_render_cosine=float((pg*rg).sum())/max(pn*rn, 1e-30))
    before = None if loss_before is None else float(loss_before.detach())
    after = None if loss_after is None else float(loss_after.detach())
    return dict(iteration=iteration, backtracks=backtracks, accepted_alpha=alpha, lambda_render=weight,
        direction_statistics_available=available,
        nominal_render_share=(weight*rn2**.5/max(pn2**.5+weight*rn2**.5, 1e-30)
                              if available else None),
        render_loss_before=before, render_loss_after=after,
        observed_render_loss_change=None if before is None or after is None else after-before,
        components_before=parts_before, components_after=parts_after, channels=channels,
        optimization_endpoint_change_rms_wu=float((endpoint_after-endpoint_before.detach()).square().sum(1).mean().sqrt()),
        interpretation='Norm share is not displacement/causal share. Direction dot delta may include gradient transforms; not physical work or an exact loss derivative. Endpoint delta is an optimizer update, not physical velocity.')


def summarize_render_influence(history, config, mpm):
    """Separate inner accepted trials from actual outer commits; retain unavailable fields."""
    def span(values):
        finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return dict(count=len(finite), minimum=min(finite), median=statistics.median(finite),
                    maximum=max(finite)) if finite else None

    attempts = [r for r in history if 'animation' in r and not r.get('held') and 'c2f_render_res' not in r]
    committed = [r for r in attempts if not r.get('outer_rejected') and
                 not r.get('null_commit') and r.get('outer_accepted', 1) and r.get('accepted', 0) > 0]
    all_steps = [s for r in attempts for s in (r.get('render_influence_steps') or [])]
    steps = [s for r in committed for s in (r.get('render_influence_steps') or [])]
    channels = sorted({k for s in steps for k in s.get('channels', {})})
    return dict(
        definitions='Observational optimizer telemetry, not causal attribution. Norm shares exclude '
                    'post-combination transforms and transport addends. Compare matched raw trajectories '
                    'with render disabled for causal evidence. Image losses do not certify physical holes/rest.',
        discretization=dict(N=config.get('_particle_count'), T=config.get('T'), dt=mpm.get('dt'),
                            dx_wu=mpm.get('dx'), loss_res=config.get('loss_res'),
                            render_res=config.get('render_res'), iters=config.get('iters')),
        render=dict(lambda_auto=config.get('lambda_auto'), surface_gs_loss=config.get('surface_gs_loss', False),
                    surface_gs_weight=config.get('surface_gs_weight'),
                    detail_height=config.get('surface_gs_detail_res')),
        windows=len(attempts), raw_history_rows=len(history), committed_windows=len(committed),
        inner_accepted_steps=sum(int(r.get('accepted', 0)) for r in attempts),
        recorded_inner_steps=len(all_steps), recorded_committed_steps=len(steps),
        steps_in_committed_windows=sum(int(r.get('accepted', 0)) for r in committed),
        step_nominal_share=span(s.get('nominal_render_share') for s in steps),
        first_iteration_nominal_share=span(r.get('g_share') for r in committed),
        adaptive_lambda=span(r.get('lambda') for r in committed),
        observed_render_loss_change=span(s.get('observed_render_loss_change') for s in steps),
        endpoint_optimizer_change_rms_wu=span(s.get('optimization_endpoint_change_rms_wu') for s in steps),
        channels={k: dict(nominal_share=span(s['channels'].get(k, {}).get('nominal_render_share') for s in steps),
                         control_delta_norm=span(s['channels'].get(k, {}).get('accepted_control_delta_norm') for s in steps))
                  for k in channels},
        last_committed_surface_components=committed[-1].get('surface_render') if committed else None,
        causal_render_ablation='not measured by this report')


def write_render_report(prefix, history, config, mpm, particle_count):
    """I/O boundary; no feedback into the numerical pipeline."""
    report = summarize_render_influence(history, dict(config, _particle_count=particle_count), mpm)
    base = Path(str(prefix)+'.render_influence')
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(base.suffix+'.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    def fmt(value):
        return ('unavailable' if value is None else
                f"{value['median']:.6g} [{value['minimum']:.6g}, {value['maximum']:.6g}], n={value['count']}")
    rows = ['# Render influence', '', report['definitions'], '',
            f"Discretization: `{json.dumps(report['discretization'])}`.", '',
            f"Committed windows: {report['committed_windows']}/{report['windows']}; "
            f"inner accepted steps in committed windows: {report['steps_in_committed_windows']} "
            f"({report['recorded_committed_steps']} with detailed telemetry).", '',
            '| Observation (median [min, max]) | Value |', '|---|---|',
            f"| Adaptive render lambda | {fmt(report['adaptive_lambda'])} |",
            f"| Nominal render direction share | {fmt(report['step_nominal_share'])} |",
            f"| First-iteration share (legacy fallback) | {fmt(report['first_iteration_nominal_share'])} |",
            f"| Observed render loss change per accepted update | {fmt(report['observed_render_loss_change'])} |",
            f"| Optimizer endpoint update RMS (wu; not velocity) | {fmt(report['endpoint_optimizer_change_rms_wu'])} |", '',
            'Matched render-off causal ablation: not measured by this report.', '']
    for name, row in report['channels'].items():
        rows.append(f"- {name}: nominal share {fmt(row['nominal_share'])}; control delta norm {fmt(row['control_delta_norm'])}.")
    components = report['last_committed_surface_components']
    if components:
        rows += ['', 'Last committed window, last inner accepted image observations:', '',
                 '| Component | Value |', '|---|---|']
        rows += [f'| {key} | {value:.8g} |' for key, value in components.items()]
        rows += ['', 'Component scales/targets differ across policies; image errors do not certify physical coverage/rest.']
    base.with_suffix(base.suffix+'.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    return report
