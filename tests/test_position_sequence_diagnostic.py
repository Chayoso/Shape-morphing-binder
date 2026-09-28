"""Statistics/scoping only; no Warp rollout or CUDA dependency."""
import torch

from scripts.probes.position_sequence_v2 import repeat_study, vector_difference


def test_vector_statistics_and_near_zero_are_explicit():
    stats = vector_difference(torch.tensor([3., 4.]), torch.tensor([0., 4.]))
    assert stats['error_l2'] == 3.
    assert stats['relative_l2'] == .75
    assert abs(stats['cosine']-.8) < 1e-14
    assert stats['strict_coordinate_failures'] == 1
    tiny = vector_difference(torch.tensor([5e-6, 0.]), torch.zeros(2))
    assert tiny['relative_l2'] is None and tiny['cosine'] is None
    assert not tiny['relative_and_direction_resolved']
    assert not tiny['strict_coordinate_pass']


def test_held_out_seed_cycle_never_changes_calibration():
    a = tuple(torch.tensor([1., 2.], requires_grad=True) for _ in range(3))
    b = tuple(torch.tensor([1., 2.], requires_grad=True) for _ in range(3))
    ref, got = tuple(2*x for x in a), tuple(2*x for x in b)
    calls = []

    def loss(outputs, kind):
        calls.append((outputs is got, kind))
        # Controlled held-out change, after 5 kinds x 8 repeats x 2 methods.
        scale = 1.5 if len(calls) > 80 and kind == 'path' else 1.
        return scale*sum((i+1)*value.sum() for i, value in enumerate(outputs))

    report = repeat_study(ref, got, a, b, loss, ('dFc', 'u', 'body'))
    assert len(calls) == 92
    assert calls[:16] == [(captured, 'path') for _ in range(8) for captured in (False, True)]
    assert [kind for _, kind in calls[80::2]] == [
        'path', 'first', 'merge', 'terminal', 'velocity', 'path']
    initial = report['calibration']['path']['dFc']
    assert initial['ordinary']['mean_gradient_rms'] == 2.
    assert initial['ordinary']['max_pair_error_rms'] == 0.
    assert len(initial['ordinary']['pairs']) == 28
    assert len(initial['cross_pairs']) == 64
    held = report['held_out'][-1]['channels']['dFc']['ordinary_vs_calibrated_mean']
    assert held['error_rms'] == 1.
    assert held['calibrated_max_pair_rms'] == 0.
    assert held['error_over_calibrated_max_pair_rms'] is None
    assert held['outside_observed_pair_rms']
    assert report['returned_forward_values_unchanged']
    assert 'passed' not in report
