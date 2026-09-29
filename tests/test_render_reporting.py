import pytest
import torch

from physmorph.pipeline.render_reporting import accepted_render_step, summarize_render_influence, write_render_report


def make_step(physics=True):
    return accepted_render_step([torch.ones(2)] if physics else None,
        [torch.ones(2)*2] if physics else None, [torch.ones(2)*.9], [torch.ones(2)], ['body'], .5,
        torch.tensor(2.), torch.tensor(1.5), {}, {}, torch.zeros(2, 3), torch.ones(2, 3)*.1, 0, 1, .01)


def test_nominal_share_is_not_direction_alignment_or_causal_motion():
    row = make_step()
    assert row['nominal_render_share'] == .5
    assert row['observed_render_loss_change'] == -.5
    assert row['channels']['body']['optimizer_render_direction_dot_delta'] < 0
    assert 'not displacement/causal' in row['interpretation']
    assert make_step(False)['nominal_render_share'] is None


def test_outer_rejected_inner_steps_are_not_reported_as_commits(tmp_path):
    row = make_step()
    history = [dict(animation=0, accepted=1, render_influence_steps=[row]),
               dict(animation=1, accepted=1, outer_rejected=1, render_influence_steps=[row]),
               dict(animation=2, accepted=0, null_commit=1), dict(animation=2, held=1),
               dict(animation=1, c2f_render_res=96)]
    result = write_render_report(tmp_path/'run', history, dict(T=20, iters=8), dict(dt=1/240, dx=.3), 300000)
    assert result['inner_accepted_steps'] == 2
    assert result['steps_in_committed_windows'] == 1
    assert result['committed_windows'] == 1
    assert result['windows'] == 3 and result['raw_history_rows'] == 5
    assert result['discretization']['N'] == 300000
    assert (tmp_path/'run.render_influence.md').exists()
    assert summarize_render_influence([], {}, {})['step_nominal_share'] is None


@torch.no_grad()
def legacy_accepted_render_step(physics, render, leaves, previous, names, weight,
                               loss_before, loss_after, parts_before, parts_after,
                               endpoint_before, endpoint_after, iteration, backtracks, alpha):
    """Frozen pre-batching reference: native reductions and Python scalar arithmetic."""
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


def reporting_case(dtypes=(torch.float32, torch.float32, torch.float64), device='cpu'):
    shapes = ((4, 3), (5, 3), (7,))
    leaves, previous, physics, render = [], [], [], []
    for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        n = torch.tensor(shape).prod().item()
        base = torch.linspace(-.37, .59, n, dtype=torch.float64).reshape(shape)
        previous.append(base.to(device=device, dtype=dtype))
        leaves.append((base+.03*(i+1)*base.cos()).to(device=device, dtype=dtype).requires_grad_())
        physics.append((base.sin()+.13).to(device=device, dtype=dtype))
        render.append((base.cos()*(-1 if i == 1 else .7)).to(device=device, dtype=dtype))
    return dict(physics=physics, render=render, leaves=leaves, previous=previous,
        names=['stress', 'body', 'surface_u'], weight=.713,
        loss_before=torch.tensor(1.+2.**-40, dtype=torch.float64, device=device),
        loss_after=torch.tensor([1.+2.**-39], dtype=torch.float64, device=device),
        parts_before={'coverage': .08, 'edge': .014}, parts_after={'coverage': .07, 'edge': .017},
        endpoint_before=torch.linspace(-.6, .9, 21, dtype=torch.float32, device=device).reshape(7, 3),
        endpoint_after=torch.linspace(-.57, .94, 21, dtype=torch.float32, device=device).reshape(7, 3),
        iteration=3, backtracks=2, alpha=.0007)


@pytest.mark.parametrize('dtypes', [
    (torch.float32,)*3, (torch.float64, torch.float32, torch.float64),
    (torch.float16, torch.bfloat16, torch.float64)])
@pytest.mark.parametrize('unavailable', [(), ('physics',), ('render',), ('physics', 'render')])
@pytest.mark.parametrize('missing', [(), ('loss_before',), ('loss_after',), ('loss_before', 'loss_after')])
def test_accepted_reporting_matches_legacy_payload_exactly(dtypes, unavailable, missing):
    case = reporting_case(dtypes)
    for name in unavailable+missing:
        case[name] = None
    actual = accepted_render_step(**case)
    assert actual == legacy_accepted_render_step(**case)
    assert actual['components_before'] is case['parts_before']
    assert actual['components_after'] is case['parts_after']
    if not missing:
        assert actual['observed_render_loss_change'] == 2.**-40


def test_reductions_keep_native_dtype_and_report_does_not_mutate_inputs():
    case = reporting_case((torch.float16, torch.bfloat16, torch.float64))
    tensors = [t for value in case.values() for t in
               (value if isinstance(value, list) else [value]) if torch.is_tensor(t)]
    before = [t.detach().clone() for t in tensors]
    actual = accepted_render_step(**case)
    assert actual == legacy_accepted_render_step(**case)
    # Promoting the inputs before reducing would change the original half norm.
    assert float(case['physics'][0].norm()) != float(case['physics'][0].double().norm())
    assert actual['channels']['stress']['physics_direction_norm'] == float(case['physics'][0].norm())
    assert all(torch.equal(t, old) for t, old in zip(tensors, before))
    assert all(t.grad is None for t in case['leaves'])


@pytest.mark.parametrize('weight', [0., .713])
def test_zero_directions_and_empty_channels_keep_legacy_semantics(weight):
    case = reporting_case()
    case['weight'] = weight
    for t in case['physics']+case['render']:
        t.zero_()
    actual = accepted_render_step(**case)
    assert actual == legacy_accepted_render_step(**case)
    assert actual['nominal_render_share'] == 0.
    assert all(c['physics_render_cosine'] == 0. for c in actual['channels'].values())
    for name in ('physics', 'render', 'leaves', 'previous', 'names'):
        case[name] = []
    actual = accepted_render_step(**case)
    assert actual == legacy_accepted_render_step(**case)
    assert actual['channels'] == {} and actual['direction_statistics_available']


def selection_record(selected=True, **fields):
    return dict(animation=19, accepted=1, rejected=2, loss=.6, d_render=.04,
        d_sil=.03, **{'lambda': .5}, render_influence_steps=[make_step()],
        window_selection=dict(selected=selected, requested_label='candidate' if selected else 'original',
            selected_label='candidate' if selected else 'original',
            donor_history_scope='Donor Adam optimization only',
            selected_work_scope='No work or Adam-step claim for selected forward',
            donor_observation=dict(loss=2., d_render=1.5, d_sil=1., **{'lambda': .5})), **fields)


def test_selected_forward_is_separate_from_donor_adam_image_delta(tmp_path):
    row = selection_record()
    result = write_render_report(tmp_path/'selected', [row], {}, {}, 300000)
    selection = result['window_selection']
    assert selection['selected_private_forwards'] == 1
    assert selection['outer_committed_selected_forwards'] == 1
    assert selection['outer_uncommitted_selected_forwards'] == 0
    assert selection['added_adam_steps'] == 0
    assert result['inner_accepted_steps'] == result['recorded_inner_steps'] == 1
    # The -0.5 donor Adam image change is neither the selected absolute error .04
    # nor the selected-minus-donor difference (.04-1.5).
    assert result['observed_render_loss_change']['median'] == -.5
    observation = selection['observations'][0]
    assert observation['donor_observation']['d_render'] == 1.5
    assert observation['selected_forward_observation']['d_render'] == .04
    assert observation['selected_forward_observation']['loss'] == .6
    assert observation['outer_committed'] is True
    assert 'DONOR' in result['definitions']
    markdown = (tmp_path/'selected.render_influence.md').read_text(encoding='utf-8')
    assert 'Added Adam steps: 0.' in markdown and 'not accepted-update deltas' in markdown
    assert '| 19 | True | True | 2.0 | 0.6 | 1.5 | 0.04 |' in markdown
    assert 'not an assertion of outer acceptance, delivered retention or causality' in markdown


def test_selection_counts_respect_outer_rejection_null_and_nonattempt_rows():
    rows = [selection_record(), selection_record(outer_rejected=1),
            selection_record(null_commit=1), selection_record(held=1),
            selection_record(c2f_render_res=96)]
    result = summarize_render_influence(rows, {}, {})
    selection = result['window_selection']
    assert selection['selection_windows'] == selection['selected_private_forwards'] == 3
    assert selection['outer_committed_selected_forwards'] == 1
    assert selection['outer_uncommitted_selected_forwards'] == 2
    assert [r['outer_committed'] for r in selection['observations']] == [True, False, False]
    assert result['inner_accepted_steps'] == 3 and result['steps_in_committed_windows'] == 1
    assert result['recorded_inner_steps'] == 3 and result['recorded_committed_steps'] == 1


def test_original_retention_is_not_counted_as_a_selected_forward():
    row = selection_record(False)
    # A refused candidate also retains the original; do not label every false
    # selection as an identity request or count its trial as a selected forward.
    refused = selection_record(False)
    refused['window_selection'].update(requested_label='failed_candidate', failures=['raw_quality'])
    result = summarize_render_influence([row, refused], {}, {})
    selection = result['window_selection']
    assert selection['selected_private_forwards'] == selection['outer_committed_selected_forwards'] == 0
    assert selection['original_result_retained_windows'] == 2
    assert all(o['selected_forward_observation'] is None for o in selection['observations'])
    assert result['recorded_inner_steps'] == 2


def test_selection_extension_preserves_existing_statistics_and_history():
    from copy import deepcopy
    history = [selection_record(), selection_record(False)]
    before = deepcopy(history)
    plain = deepcopy(history)
    for row in plain:
        row.pop('window_selection')
    legacy = summarize_render_influence(plain, {}, {})
    result = summarize_render_influence(history, {}, {})
    assert {k: v for k, v in result.items() if k not in ('window_selection', 'definitions')} == {
        k: v for k, v in legacy.items() if k != 'definitions'}
    assert result['definitions'].startswith(legacy['definitions'])
    assert history == before
    assert 'window_selection' not in legacy


def test_missing_donor_observations_are_not_fabricated_from_selected_values():
    row = selection_record()
    row['window_selection'].pop('donor_observation')
    result = summarize_render_influence([row], {}, {})
    observation = result['window_selection']['observations'][0]
    assert all(v is None for v in observation['donor_observation'].values())
    assert observation['selected_forward_observation']['loss'] == .6


def test_default_empty_report_payload_is_unchanged():
    assert summarize_render_influence([], {}, {}) == dict(
        definitions='Observational optimizer telemetry, not causal attribution. Norm shares exclude '
                    'post-combination transforms and transport addends. Compare matched raw trajectories '
                    'with render disabled for causal evidence. Image losses do not certify physical holes/rest.',
        discretization=dict(N=None, T=None, dt=None, dx_wu=None, loss_res=None, render_res=None, iters=None),
        render=dict(lambda_auto=None, surface_gs_loss=False, surface_gs_weight=None, detail_height=None),
        windows=0, raw_history_rows=0, committed_windows=0, inner_accepted_steps=0,
        recorded_inner_steps=0, recorded_committed_steps=0, steps_in_committed_windows=0,
        step_nominal_share=None, first_iteration_nominal_share=None, adaptive_lambda=None,
        observed_render_loss_change=None, endpoint_optimizer_change_rms_wu=None, channels={},
        last_committed_surface_components=None, causal_render_ablation='not measured by this report')
