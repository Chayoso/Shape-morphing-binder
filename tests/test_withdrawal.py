"""Owned passive inputs and full-state closure; CPU unit tests, no pipeline."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.traj import Trajectory
from physmorph.mpm.withdrawal import OwnedWithdrawal
from test_boundary_packet import _trajectory


def owned_tensor(value):
    return wp.to_torch(value).detach().clone()


def direct_withdrawal(head, step):
    """Independent oracle: existing fixture, overwritten full physical state."""
    direct = _trajectory(pin_slip=bool(head.pin_mode))
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        wp.copy(getattr(direct, name)[0], getattr(head, name)[step])
    for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol', 'layer_mask', 'layer_nrm',
                 'layer_nbr', 'layer_w', 'layer_ug', 'layer_g', 'bond_nbr', 'bond_rest', 'bond_frag'):
        wp.copy(getattr(direct, name), getattr(head, name))
    direct.prm = deepcopy(head.prm)
    direct.gate_n0 = head.gate_n0
    direct.layer_frac, direct.layer_frac_u = head.layer_frac, head.layer_frac_u
    direct.layer_inv_depth, direct.frag_thr = head.layer_inv_depth, head.frag_thr
    for value in direct.dFc_seq:
        value.zero_()
    direct.layer_u.zero_()
    direct.body_control = None
    return direct


@pytest.mark.parametrize('step,persistent', [(0, False), (20, True)])
def test_complete_nonlinear_state_and_frozen_policy_match_direct_withdrawal(step, persistent, monkeypatch):
    head = _trajectory(pin_slip=True)
    head.rollout()
    # Prepared policies may change after construction; capture their actual values.
    head.layer_ug.fill_(.37)
    head.layer_frac = .017
    head.layer_frac_u = .043
    head.layer_inv_depth = 6.123456789
    head.frag_step.fill_(99.)  # These are last-step scratch, never permanent input.
    head.ncount_b.fill_(999.)
    direct = direct_withdrawal(head, step)
    with monkeypatch.context() as patch:
        patch.setattr(Trajectory, 'step', lambda *_: pytest.fail('capture/build executed dynamics'))
        snapshot = OwnedWithdrawal.capture(head, step)
        replay = snapshot.trajectory(persistent=persistent)
    meta = snapshot.metadata()
    assert meta['T'] == replay.T == 20
    assert replay.persistent == persistent and replay.graph is None
    assert meta['step'] == step and meta['source_body_control']
    assert replay.body_control is None
    assert replay.layer_frac == head.layer_frac and replay.layer_frac_u == head.layer_frac_u
    assert replay.layer_inv_depth == head.layer_inv_depth and replay.frag_thr == 2.5
    assert meta['prm']['gate_n0'] == 0. and meta['gate_n0'] > 0
    assert replay.gate_n0 == replay.prm.gate_n0 == head.gate_n0
    assert head.prm.gate_n0 == 0.  # Resolving replay normalization cannot mutate source prm.
    assert replay.pin_mode == 1 and replay.layer_F and replay.track_geom
    for state in ('v', 'C', 'F', 'Fg'):
        torch.testing.assert_close(owned_tensor(getattr(replay, state)[0]),
                                   owned_tensor(getattr(head, state)[step]), rtol=0, atol=0)
    assert torch.count_nonzero(owned_tensor(replay.C[0])) > 0
    assert torch.count_nonzero(owned_tensor(replay.v[0])) > 0
    assert not torch.equal(owned_tensor(replay.Fp), torch.eye(3).expand(head.N, -1, -1))
    for t in range(replay.T):
        assert torch.count_nonzero(owned_tensor(replay._dfc(t))) == 0
    assert torch.count_nonzero(owned_tensor(replay.layer_u)) == 0
    assert not replay.x[0].requires_grad and not replay.Fp.requires_grad
    assert torch.count_nonzero(owned_tensor(head._dfc(0))) > 0
    assert torch.count_nonzero(owned_tensor(head.layer_u)) > 0
    original_head = {name: owned_tensor(getattr(head, name)[step]) for name in ('x', 'v', 'C', 'F', 'Fg')}
    replay.run()
    direct.run()
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        for t in range(21):
            torch.testing.assert_close(owned_tensor(getattr(replay, name)[t]),
                                       owned_tensor(getattr(direct, name)[t]), rtol=0, atol=0)
        torch.testing.assert_close(owned_tensor(getattr(head, name)[step]), original_head[name], rtol=0, atol=0)
    torch.testing.assert_close(owned_tensor(replay.frag_step), owned_tensor(direct.frag_step), rtol=0, atol=0)
    assert torch.max(owned_tensor(replay.frag_step)) <= 1.
    assert torch.count_nonzero(owned_tensor(replay.x[1])-owned_tensor(replay.xu[1])) > 0
    pins = owned_tensor(replay.pin) > .5
    assert torch.equal(owned_tensor(replay.x[-1])[pins], owned_tensor(replay.x[0])[pins])
    assert torch.count_nonzero(owned_tensor(replay.v[-1])[pins]) == 0


def test_archive_roundtrip_deep_ownership_and_private_replays(tmp_path):
    head = _trajectory(pin_slip=True)
    head.rollout()
    snapshot = OwnedWithdrawal.capture(head, head.T)
    before, meta = snapshot.arrays(), snapshot.metadata()
    path = tmp_path/'withdrawal.npz'
    np.savez(path, **before, __metadata__=json.dumps(meta, allow_nan=False))
    with np.load(path, allow_pickle=False) as archive:
        loaded_meta = json.loads(str(archive['__metadata__']))
        loaded = {k: archive[k].copy() for k in archive.files if k != '__metadata__'}
    restored = OwnedWithdrawal.from_arrays(loaded, loaded_meta, device='cpu')
    assert restored.metadata()['captured_device'] == 'cpu'
    # Neither source storage, archived arrays nor public accessors may alias owner.
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        for value in getattr(head, name):
            value.zero_()
    for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol', 'layer_mask', 'layer_nrm',
                 'layer_nbr', 'layer_w', 'layer_ug', 'layer_g', 'bond_nbr', 'bond_rest', 'bond_frag'):
        getattr(head, name).zero_()
    head.prm.dt = .3
    head.prm.grid_min = [10., 10., 10.]
    for mapping in (loaded, snapshot.arrays(), restored.arrays()):
        for value in mapping.values():
            value.fill(0)
    loaded_meta['prm']['grid_min'][0] = 10.
    accessor = restored.metadata()
    accessor['prm']['dt'] = .5
    accessor['layer_frac'] = .9
    for actual in (snapshot.arrays(), restored.arrays()):
        for key in before:
            np.testing.assert_array_equal(actual[key], before[key])
    assert restored.metadata()['prm']['dt'] == snapshot.metadata()['prm']['dt'] == .002
    left, right = snapshot.trajectory(), restored.trajectory()
    left.run(); right.run()
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        torch.testing.assert_close(owned_tensor(getattr(left, name)[20]),
                                   owned_tensor(getattr(right, name)[20]), rtol=0, atol=0)
        getattr(left, name)[0].zero_()
    for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol', 'layer_mask', 'layer_ug',
                 'layer_nrm', 'layer_nbr', 'layer_w', 'layer_g', 'bond_nbr', 'bond_rest', 'bond_frag'):
        getattr(left, name).zero_()
    left.prm.dt = 1.
    fresh = restored.trajectory()
    assert fresh.prm.dt == right.prm.dt == .002
    for owner in (snapshot, restored):
        for key, value in owner.arrays().items():
            np.testing.assert_array_equal(value, before[key])
    for name, key in (('Fp', 'Fp'), ('pin', 'pin'), ('layer_ug', 'layer_ug'), ('bond_frag', 'bond_frag')):
        np.testing.assert_array_equal(owned_tensor(getattr(right, name)).numpy(), before[key])


def test_support_reference_not_reestimated_after_captured_geometry_changes(monkeypatch):
    head = _trajectory(pin_slip=False)
    head.rollout()
    snapshot = OwnedWithdrawal.capture(head, 20)
    with monkeypatch.context() as patch:
        patch.setattr('physmorph.mpm.traj.nominal_support',
                      lambda *_: pytest.fail('resolved source normalization was recomputed'))
        replay = snapshot.trajectory()
    assert replay.gate_n0 == head.gate_n0 and replay.pin_mode == 0
    np.testing.assert_array_equal(owned_tensor(replay.vol).numpy(), owned_tensor(head.vol).numpy())


def test_without_optional_policies_still_owns_accumulated_state():
    head = _trajectory(extras=False)
    head.rollout()
    snapshot = OwnedWithdrawal.capture(head, 20)
    replay = snapshot.trajectory()
    assert not replay.layer and not replay.bonds and replay.Fg is None and not replay.gate
    assert replay.body_control is None and replay.T == 20
    torch.testing.assert_close(owned_tensor(replay.F[0]), owned_tensor(head.F[20]), rtol=0, atol=0)
    assert snapshot.metadata()['gate_n0'] is None
    assert not snapshot.arrays()['layer_mask'].any()
    replay.run()


@pytest.mark.parametrize('step', [-1, 21, True, 1.5])
def test_invalid_capture_step_rejected(step):
    with pytest.raises(ValueError, match='step'):
        OwnedWithdrawal.capture(_trajectory(extras=False), step)


@pytest.mark.parametrize('alter', [
    lambda a, m: m.update(T=0),
    lambda a, m: m.update(step=21),
    lambda a, m: m.update(gate_n0=0),
    lambda a, m: m.update(layer_K=0),
    lambda a, m: m.update(pin_mode=2),
    lambda a, m: m['prm'].pop('smoothing'),
    lambda a, m: a.pop('C0'),
    lambda a, m: a.update(dFc=np.zeros((m['N'], 3, 3), np.float32)),
    lambda a, m: a.update(layer_nbr=a['layer_nbr'].astype(np.float32)),
    lambda a, m: a['layer_nbr'].__setitem__((0, 0), -1),
    lambda a, m: a['bond_nbr'].__setitem__((0, 0), m['N']),
    lambda a, m: a.update(Fp=a['Fp'].reshape(-1, 9)),
])
def test_invalid_archive_schema_and_policy_cannot_silently_change_replay(alter):
    snapshot = OwnedWithdrawal.capture(_trajectory(), 0)
    arrays, meta = snapshot.arrays(), snapshot.metadata()
    alter(arrays, meta)
    with pytest.raises(ValueError):
        OwnedWithdrawal.from_arrays(arrays, meta)


def test_context_mismatch_rejected_before_any_array_conversion(monkeypatch):
    snapshot = OwnedWithdrawal.capture(_trajectory(extras=False), 0)
    import physmorph.mpm.withdrawal as module
    with monkeypatch.context() as patch:
        patch.setattr(module, 'is_cuda_execution', lambda: True)
        patch.setattr(module, 'to_array', lambda *_a, **_k: pytest.fail('context failure used a conversion'))
        with pytest.raises(RuntimeError, match='context'):
            snapshot.trajectory()
        with pytest.raises(RuntimeError, match='context'):
            snapshot.arrays()
    class FakeCudaDevice:
        is_cuda = True
        ordinal = 0
    arrays, metadata = snapshot.arrays(), snapshot.metadata()
    with monkeypatch.context() as patch:
        patch.setattr(wp, 'get_device', lambda *_: FakeCudaDevice())
        patch.setattr(module, 'to_array', lambda *_a, **_k: pytest.fail('GPU fallback reached host conversion'))
        with pytest.raises(RuntimeError, match='context'):
            OwnedWithdrawal.from_arrays(arrays, metadata, device='cuda:0')
