"""Accepted-window checkpoint pairing must survive rejected attempts."""
import json
import numpy as np
import pytest

from scripts.probes.boundary_replay import Capture, load_packet, save_packet, check_original_step


def candidate(attempt, marker, *, boundary=False, packet=False):
    return dict(attempt=attempt,
                boundary=({k: np.full((2, 3), marker, np.float32)
                           for k in ('previous', 'raw', 'promoted')} if boundary else None),
                packet=(dict(meta={'T': 20}, arrays={'x0': np.full((2, 3), marker, np.float32)})
                        if packet else None))


def test_rejected_next_attempt_is_not_admitted_or_used_as_boundary(tmp_path):
    c = Capture(tmp_path, (1, 2))
    c.candidate = candidate(1, 10, boundary=True)
    c.commit(0, None, None, None, {'frame_end': 21})
    c.candidate = candidate(2, 999, boundary=True, packet=True)
    c.commit(1, None, None, None, {'outer_rejected': 1})
    assert c.accepted == 1 and c.pending['attempt'] == 1 and c.saved == []
    c.candidate = candidate(3, 20, boundary=True, packet=True)
    c.commit(2, None, None, None, {'frame_end': 41})
    packet = load_packet(tmp_path / 'boundary_001.npz')
    assert packet['meta']['T'] == 20
    assert packet['meta']['boundary_attempt'] == 1
    assert packet['meta']['next_attempt'] == 3
    np.testing.assert_array_equal(packet['arrays']['raw'], 10)
    np.testing.assert_array_equal(packet['arrays']['x0'], 20)
    assert c.pending['attempt'] == 3 and c.pending['commit'] == 2
    c.candidate = candidate(4, 30, packet=True)
    c.commit(3, None, None, None, {'frame_end': 61})
    following = load_packet(tmp_path / 'boundary_002.npz')
    np.testing.assert_array_equal(following['arrays']['raw'], 20)
    assert [p['boundary_commit'] for p in c.saved] == [1, 2]


def test_accepted_callback_cannot_reuse_stale_attempt(tmp_path):
    c = Capture(tmp_path, (1,))
    c.candidate = candidate(1, 10, boundary=True)
    with pytest.raises(RuntimeError, match='observer'):
        c.commit(1, None, None, None, {'frame_end': 21})
    assert not list(tmp_path.iterdir())


def test_packet_roundtrip_is_pickle_free_owned_and_exclusive(tmp_path):
    path = tmp_path / 'packet.npz'
    value = np.arange(12, dtype=np.float32).reshape(4, 3)
    packet = dict(meta={'T': 20, 'layer': False}, arrays={'x0': value})
    save_packet(path, packet)
    value[:] = -1
    loaded = load_packet(path)
    np.testing.assert_array_equal(loaded['arrays']['x0'], np.arange(12).reshape(4, 3))
    assert json.dumps(loaded['meta']) == json.dumps(packet['meta'])
    with pytest.raises(FileExistsError):
        save_packet(path, packet)


def test_real_optimizer_observer_captures_actual_next_accepted_step(tmp_path, monkeypatch):
    from physmorph.pipeline import runner, PipelineConfig
    from physmorph.mpm.state import MPMParams
    src = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (src * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=3, animations=2, loss_res=12, render_views=2,
                         render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
                         commit_pic=True, commit_pic_objective=True, body_ctrl=True,
                         lambda_auto=.3, w_kin=0., w_ctrl=0., w_box=0.)
    c = Capture(tmp_path, (1,))
    monkeypatch.setattr(runner, 'optimize_window', c.wrap(runner.optimize_window))
    result = runner.run_pipeline(src, target, MPMParams(dx=1., nx=32, ny=32, nz=32),
                                 cfg, log=lambda *_: None, on_commit=c.commit)
    assert c.accepted == 2 and len(c.saved) == 1
    packet = load_packet(tmp_path / 'boundary_001.npz')
    p = packet['arrays']
    np.testing.assert_array_equal(p['x0'], p['promoted'])
    np.testing.assert_array_equal(p['previous'], result['frames'][2])
    assert np.max(np.abs(p['raw'] - p['promoted'])) > 1e-7
    np.testing.assert_array_equal(p['original_x1'], result['frames'][4])
    assert packet['meta']['T'] == 3
    assert packet['meta']['has_body_control']


@pytest.mark.parametrize('field', ['x1', 'pre_layer', 'v1', 'F1', 'C1'])
def test_closure_is_invariant_to_consistent_length_and_time_units(field):
    import torch
    ref = torch.tensor([.1, 2., -.3], dtype=torch.float64)
    got = ref + torch.tensor([.0001, -.0002, .00001], dtype=torch.float64)
    dt, dx = 1 / 240, .3
    length_factor, time_factor = 1000., .001
    units = dict(x1=length_factor, pre_layer=length_factor,
                 v1=length_factor / time_factor, F1=1., C1=1 / time_factor)
    first = check_original_step(got, ref, field, dt, dx)
    second = check_original_step(got * units[field], ref * units[field], field,
                                 dt * time_factor, dx * length_factor)
    assert first['passed'] == second['passed']
    assert first['max_tolerance_ratio'] == pytest.approx(second['max_tolerance_ratio'], rel=1e-10)


@pytest.mark.parametrize('field,factor', [('x1', .3), ('pre_layer', .3),
                                          ('v1', .3 * 240), ('F1', 1.), ('C1', 240.)])
def test_closure_rejects_same_dimensionless_state_error_in_every_field(field, factor):
    import torch
    reference = torch.zeros(3, dtype=torch.float64)
    assert not check_original_step(reference + 1e-3 * factor, reference, field, 1 / 240, .3)['passed']
