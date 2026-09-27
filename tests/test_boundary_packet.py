"""Accepted-buffer ownership and first-step reconstruction on Warp CPU only."""
import json

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory
from scripts.probes.boundary_packet import capture_step, replay_step


def _trajectory(*, pin_slip=False, extras=True):
    rng = np.random.default_rng(251)
    x = np.stack(np.meshgrid(*([np.arange(3) * .15] * 3), indexing='ij'), -1)
    x = x.reshape(-1, 3).astype(np.float32) - .15
    x += rng.uniform(-.008, .008, x.shape).astype(np.float32)
    x[-1] = [2., 0., 0.]  # Current-density bond activation, without frozen frag.
    n, T = len(x), 20
    prm = MPMParams(dx=.5, dt=.002, nx=16, ny=16, nz=16,
                    grid_min=(-4., -4., -4.), f_ext=(0., -.02, 0.),
                    drag=.04, smoothing=.97, eta_sym=1, eta_mode=1,
                    gate_r_lo=.1 if extras else 0., gate_r_hi=.8 if extras else 0.)
    ids = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()
    F = ids + rng.normal(0., .002, ids.shape).astype(np.float32)
    Fp = ids + rng.normal(0., .001, ids.shape).astype(np.float32)
    C = rng.normal(0., .008, ids.shape).astype(np.float32)
    v = rng.normal(0., .01, x.shape).astype(np.float32)
    dfc = [wp.array(np.full_like(ids, .002 + t * .0001), dtype=wp.mat33,
                    device='cpu') for t in range(T)]
    pin = np.zeros(n, np.float32)
    pin[[0, 8]] = 1
    kwargs = {}
    if extras:
        dist = np.linalg.norm(x[:, None] - x[None, :], axis=2)
        nbr = np.argsort(dist, axis=1)[:, 1:5].astype(np.int32)
        rest = np.take_along_axis(dist, nbr, axis=1).astype(np.float32)
        mask = np.ones(n, np.float32); mask[2] = 0
        nrm = np.tile(np.array([0., 1., 0.], np.float32), (n, 1))
        w = np.full(nbr.shape, .25, np.float32); w[2] = 0
        g = rng.normal(0., .02, (n, 4, 3)).astype(np.float32)
        ug = np.linspace(.1, .9, n, dtype=np.float32)
        frag = np.zeros(n, np.float32); frag[1] = 1
        body = rng.normal(0., .001, (2 * n, 3)).astype(np.float32)
        kwargs = dict(
            bonds=(nbr, rest, frag, 2.5),
            layer=(mask, nrm, nbr, w, .6 / T, g, .15, ug),
            layer_u=wp.array(np.linspace(-.003, .004, n, dtype=np.float32),
                             dtype=wp.float32, device='cpu'),
            body_control=wp.array(body, dtype=wp.vec3, device='cpu'))
    return Trajectory(
        x, np.linspace(.7, 1.3, n, dtype=np.float32),
        np.linspace(180., 220., n, dtype=np.float32),
        np.linspace(90., 110., n, dtype=np.float32), prm, T,
        Fp=Fp, F0=F, C0=C, v0=v, dFc=dfc,
        eta=np.linspace(1., 3., n, dtype=np.float32), pin=pin,
        pin_slip=pin_slip, device='cpu', requires_grad=False, persistent=True,
        vol0=np.linspace(.003, .005, n, dtype=np.float32),
        track_geom=extras, Fg0=F.copy() if extras else None, **kwargs)


def _full_rollout():
    tr = _trajectory(pin_slip=True)
    tr.step(0)
    activity = wp.to_torch(tr.frag_step).clone()
    counts = wp.to_torch(tr.ncount_b).clone()
    for step in range(1, tr.T):
        tr.step(step)
    return tr, activity, counts


def _assert_closure(packet, result):
    for name in ('x1', 'v1', 'F1', 'C1', 'pre_layer'):
        torch.testing.assert_close(result[name],
                                   torch.from_numpy(packet['arrays']['original_' + name]),
                                   rtol=2e-6, atol=2e-7)
        assert not result[name].requires_grad


@pytest.mark.parametrize('pin_slip', [False, True])
def test_full_rollout_capture_replays_actual_first_step_and_original_horizon(pin_slip, monkeypatch):
    tr = _trajectory(pin_slip=pin_slip)
    tr.step(0)
    activity = wp.to_torch(tr.frag_step).clone()
    counts = wp.to_torch(tr.ncount_b).clone()
    assert torch.any(activity == 0) and torch.any(activity == 1)
    for step in range(1, tr.T):
        tr.step(step)
    # The final scratch buffers are not valid first-step evidence.
    tr.frag_step.fill_(99.)
    tr.ncount_b.fill_(123.)
    with monkeypatch.context() as patch:
        patch.setattr(Trajectory, 'step', lambda *_: pytest.fail('capture executed a step'))
        packet = capture_step(tr)
    assert packet['meta']['T'] == 20
    assert packet['meta']['prm']['gate_n0'] == 0
    assert packet['meta']['gate_n0'] > 0
    assert packet['meta']['pin_mode'] == int(pin_slip)
    assert packet['arrays']['body_control'].shape == (2 * tr.N, 3)
    assert packet['arrays']['layer_nbr'].shape == (tr.N, 4)
    assert packet['arrays']['layer_g'].shape == (tr.N, 4, 3)
    called = []
    original_step = Trajectory.step

    def replay_once(rebuilt, t):
        called.append(t)
        assert rebuilt.T == 20 and not rebuilt.persistent
        assert not rebuilt.x[0].requires_grad
        return original_step(rebuilt, t)

    with monkeypatch.context() as patch:
        patch.setattr(Trajectory, 'step', replay_once)
        result = replay_step(packet)
    assert called == [0]
    _assert_closure(packet, result)
    torch.testing.assert_close(result['bond_active'], activity, rtol=0, atol=0)
    torch.testing.assert_close(result['neighbor_count'], counts, rtol=0, atol=0)
    torch.testing.assert_close(result['Fg1'], torch.from_numpy(packet['arrays']['original_Fg1']))
    assert torch.count_nonzero(result['x1'] - result['pre_layer']) > 0
    pinned = torch.from_numpy(packet['arrays']['pin'] > .5)
    assert torch.equal(result['x1'][pinned], result['x0'][pinned])
    # A T=1 reconstruction would lose both pulse modes and change 1/T channels.
    assert packet['meta']['layer_frac_u'] == pytest.approx(1 / 20)
    assert torch.count_nonzero(result['C1']) > 0


def test_packet_npz_json_roundtrip_owns_every_input_output_and_metadata(tmp_path):
    tr, _, _ = _full_rollout()
    packet = capture_step(tr)
    expected = {key: value.copy() for key, value in packet['arrays'].items()}
    expected_meta = json.loads(json.dumps(packet['meta']))
    path = tmp_path / 'step.npz'
    np.savez(path, **packet['arrays'], __meta__=json.dumps(packet['meta']))
    with np.load(path, allow_pickle=False) as archive:
        loaded = {'meta': json.loads(str(archive['__meta__'])),
                  'arrays': {key: archive[key].copy() for key in archive.files if key != '__meta__'}}
    # Accepted eval buffers, caller controls, material data and dataclass are reusable.
    for name in ('x', 'v', 'F', 'C', 'Fg', 'xu'):
        for value in getattr(tr, name):
            value.zero_()
    for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol', 'body_control',
                 'layer_mask', 'layer_nrm', 'layer_nbr', 'layer_w', 'layer_u',
                 'layer_ug', 'layer_g', 'bond_nbr', 'bond_rest', 'bond_frag'):
        getattr(tr, name).zero_()
    for value in tr.dFc_seq:
        value.zero_()
    tr.prm.dt = .1
    assert packet['meta'] == loaded['meta'] == expected_meta
    for key, value in expected.items():
        np.testing.assert_array_equal(packet['arrays'][key], value)
        np.testing.assert_array_equal(loaded['arrays'][key], value)
    result = replay_step(loaded)
    _assert_closure(loaded, result)
    # No replay output aliases either the packet's host memory or another output.
    result['x0'].zero_()
    result['pre_layer'].zero_()
    np.testing.assert_array_equal(loaded['arrays']['x0'], expected['x0'])
    torch.testing.assert_close(result['x1'], torch.from_numpy(expected['original_x1']))


def test_override_recomputes_density_activation_without_mutating_frozen_packet():
    tr, activity, _ = _full_rollout()
    packet = capture_step(tr)
    before = {key: value.copy() for key, value in packet['arrays'].items()}
    changed = torch.from_numpy(packet['arrays']['x0'].copy())
    changed[-1] = changed[3]  # A formerly isolated unpinned particle joins the body.
    result = replay_step(packet, changed)
    assert activity[-1] == 1 and result['bond_active'][-1] == 0
    assert result['neighbor_count'][-1] > 2.5
    assert torch.equal(result['x0'], changed)
    for key, value in before.items():
        np.testing.assert_array_equal(packet['arrays'][key], value)
    # Compare to a directly constructed full-horizon trajectory with the same inputs.
    direct = _trajectory(pin_slip=True)
    wp.to_torch(direct.x[0]).copy_(changed)
    direct.step(0)
    torch.testing.assert_close(result['x1'], wp.to_torch(direct.x[1]))
    torch.testing.assert_close(result['v1'], wp.to_torch(direct.v[1]))


def test_absent_optional_fields_and_output_ownership():
    tr = _trajectory(extras=False)
    tr.rollout()
    packet = capture_step(tr)
    assert not packet['meta']['layer_present']
    assert not packet['meta']['bonds_present']
    assert not packet['meta']['track_geom']
    assert not packet['meta']['has_body_control']
    assert not packet['arrays']['layer_mask'].any()
    result = replay_step(packet)
    _assert_closure(packet, result)
    assert 'bond_active' not in result and 'Fg1' not in result
    assert torch.equal(result['x1'], result['pre_layer'])
    result['pre_layer'].zero_()
    assert torch.count_nonzero(result['x1']) > 0
    with pytest.raises(ValueError, match='shape'):
        replay_step(packet, np.zeros((1, 3), np.float32))
    with pytest.raises(RuntimeError, match='cuda_execution'):
        replay_step(packet, device='cuda')
