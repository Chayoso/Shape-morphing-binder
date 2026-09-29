"""Accepted-step render influence, with explicit non-causal interpretations."""
import json
import math
import os
from pathlib import Path
import statistics

import torch


@torch.no_grad()
def accepted_render_step(physics, render, leaves, previous, names, weight,
                         loss_before, loss_after, parts_before, parts_after,
                         endpoint_before, endpoint_after, iteration, backtracks, alpha):
    observations, channel_offsets = [], []
    available = physics is not None and render is not None
    for i, (p, old, name) in enumerate(zip(leaves, previous, names)):
        delta = p.detach()-old
        channel_offsets.append((name, len(observations)))
        if not available:
            observations.append(delta.norm())
            continue
        pg, rg = physics[i], render[i]
        observations.extend((pg.norm(), rg.norm(), delta.norm(),
                             (rg*delta).sum(), (pg*rg).sum()))
    before_index = None if loss_before is None else len(observations)
    if loss_before is not None:
        observations.append(loss_before.detach())
    after_index = None if loss_after is None else len(observations)
    if loss_after is not None:
        observations.append(loss_after.detach())
    observations.append((endpoint_after-endpoint_before.detach()).square().sum(1).mean().sqrt())
    # Reductions retain their original dtype. Stack promotes (including float64)
    # only the finished scalars; one host transfer supplies all Python reporting.
    # Device metadata is host-known; gathering also preserves mixed-device callers.
    device = next((value.device for value in observations if value.is_cuda), observations[0].device)
    values = torch.stack([value.to(device=device).reshape(()) for value in observations]).cpu().tolist()
    channels = {}
    pn2, rn2 = 0., 0.
    for name, offset in channel_offsets:
        if not available:
            channels[name] = dict(accepted_control_delta_norm=values[offset],
                                  direction_statistics=None)
            continue
        pn, rn, delta_norm, render_dot_delta, physics_dot_render = values[offset:offset+5]
        pn2 += pn*pn; rn2 += rn*rn
        channels[name] = dict(physics_direction_norm=pn, render_direction_norm=rn,
            nominal_render_share=weight*rn/max(pn+weight*rn, 1e-30),
            accepted_control_delta_norm=delta_norm,
            optimizer_render_direction_dot_delta=render_dot_delta,
            weighted_optimizer_render_direction_dot_delta=weight*render_dot_delta,
            physics_render_cosine=physics_dot_render/max(pn*rn, 1e-30))
    before = None if before_index is None else values[before_index]
    after = None if after_index is None else values[after_index]
    return dict(iteration=iteration, backtracks=backtracks, accepted_alpha=alpha, lambda_render=weight,
        direction_statistics_available=available,
        nominal_render_share=(weight*rn2**.5/max(pn2**.5+weight*rn2**.5, 1e-30)
                              if available else None),
        render_loss_before=before, render_loss_after=after,
        observed_render_loss_change=None if before is None or after is None else after-before,
        components_before=parts_before, components_after=parts_after, channels=channels,
        optimization_endpoint_change_rms_wu=values[-1],
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
    report = dict(
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
    selections = [r for r in attempts if r.get('window_selection') is not None]
    if selections:
        selected = [r for r in selections if r['window_selection'].get('selected') is True]
        committed_ids = {id(r) for r in committed}
        scope = ('Direction/work statistics, accepted-update image-loss deltas and optimizer endpoint '
                 'changes describe recorded Adam updates. In selected windows these are DONOR '
                 'optimization evidence, not measurements of the fresh private forward. '
                 'Selection is not an additional Adam step or causal motion attribution.')
        report['definitions'] += ' ' + scope
        observations = []
        for r in selections:
            selection = r['window_selection']
            donor = selection.get('donor_observation') or {}
            observations.append(dict(animation=r['animation'], selected=selection.get('selected') is True,
                outer_committed=id(r) in committed_ids,
                requested_label=selection.get('requested_label'), selected_label=selection.get('selected_label'),
                donor_history_scope=selection.get('donor_history_scope'),
                selected_work_scope=selection.get('selected_work_scope'),
                donor_observation={k: donor.get(k) for k in ('loss', 'd_render', 'd_sil', 'lambda')},
                selected_forward_observation=({k: r.get(k) for k in ('loss', 'd_render', 'd_sil', 'lambda')}
                                              if selection.get('selected') is True else None)))
        report['window_selection'] = dict(
            selection_windows=len(selections), selected_private_forwards=len(selected),
            outer_committed_selected_forwards=sum(id(r) in committed_ids for r in selected),
            outer_uncommitted_selected_forwards=sum(id(r) not in committed_ids for r in selected),
            original_result_retained_windows=len(selections)-len(selected),
            added_adam_steps=0, optimizer_observation_scope=scope,
            selected_observation_scope='Fresh private head-forward values, separate from donor update deltas; '
                                       'not an assertion of outer acceptance, delivered retention or causality.',
            observations=observations)
    return report


def write_render_report(prefix, history, config, mpm, particle_count, *, reserve_bytes=None):
    """I/O boundary; no feedback into the numerical pipeline."""
    report = summarize_render_influence(history, dict(config, _particle_count=particle_count), mpm)
    base = Path(str(prefix)+'.render_influence')
    base.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(report, indent=2).replace('\n', os.linesep).encode('utf-8')
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
    if 'window_selection' in report:
        selection = report['window_selection']
        rows += [f"Selected private forwards: {selection['selected_private_forwards']}; "
                 f"outer committed: {selection['outer_committed_selected_forwards']}; "
                 f"outer uncommitted: {selection['outer_uncommitted_selected_forwards']}. "
                 f"Original result retained: {selection['original_result_retained_windows']}. "
                 'Added Adam steps: 0.', '', selection['optimizer_observation_scope'], '',
                 'Fresh selected-head observations and donor observations (not accepted-update deltas):', '',
                 '| Attempt (zero-based) | Selected | Outer committed | Donor loss | Selected loss | Donor render | Selected render |',
                 '|---|---|---|---|---|---|---|']
        for observation in selection['observations']:
            donor = observation['donor_observation']
            fresh = observation['selected_forward_observation'] or {}
            rows.append(f"| {observation['animation']} | {observation['selected']} | "
                        f"{observation['outer_committed']} | {donor.get('loss')} | {fresh.get('loss')} | "
                        f"{donor.get('d_render')} | {fresh.get('d_render')} |")
        rows += ['', selection['selected_observation_scope'], '']
    for name, row in report['channels'].items():
        rows.append(f"- {name}: nominal share {fmt(row['nominal_share'])}; control delta norm {fmt(row['control_delta_norm'])}.")
    components = report['last_committed_surface_components']
    if components:
        component_scope = ('Last committed window, last inner accepted image observations '
                           '(donor evidence if that window selected a private forward):'
                           if 'window_selection' in report else
                           'Last committed window, last inner accepted image observations:')
        rows += ['', component_scope, '',
                 '| Component | Value |', '|---|---|']
        rows += [f'| {key} | {value:.8g} |' for key, value in components.items()]
        rows += ['', 'Component scales/targets differ across policies; image errors do not certify physical coverage/rest.']
    markdown_payload = ('\n'.join(rows)+'\n').replace('\n', os.linesep).encode('utf-8')
    if reserve_bytes is not None:
        reserve_bytes(len(json_payload)+len(markdown_payload))
    base.with_suffix(base.suffix+'.json').write_bytes(json_payload)
    base.with_suffix(base.suffix+'.md').write_bytes(markdown_payload)
    return report
