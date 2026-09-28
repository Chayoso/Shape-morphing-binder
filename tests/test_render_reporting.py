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
