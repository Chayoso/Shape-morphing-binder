"""Window geometry preserves material history and geometric covariance."""
from copy import deepcopy

import numpy as np
import pytest

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.preparation_geometry import prepare_bonds, prepare_layer_geometry


def test_detached_bond_rows_keep_history_until_reconnection():
    x = np.array([[i*.2, j*.2, k*.2] for i in range(4) for j in range(3)
                  for k in range(3)], dtype=np.float32)
    detached = np.array([[6., 0., 0.], [6.1, 0., 0.]], dtype=np.float32)
    x = np.concatenate([x, detached])
    neighbors = ((np.arange(len(x))+1) % len(x))[:, None].astype(np.int32)
    prm = MPMParams(dx=.5, grid_min=(-2., -2., -2.), nx=24, ny=16, nz=16)
    history = np.full((len(x), 1), 123., dtype=np.float32)
    initial = history.copy()
    rest, fragmented = prepare_bonds(x, neighbors, history, prm)
    assert fragmented[-2:].all() and not fragmented[:-2].any()
    assert np.array_equal(rest[-2:], history[-2:])
    assert (rest[:-2] < 123.).all()
    assert np.array_equal(history, initial)
    x[-2:, 0] -= 5.
    reconnected, fragmented = prepare_bonds(x, neighbors, rest, prm)
    assert not fragmented.any() and (reconnected[-2:] < 123.).all()
    assert np.array_equal(rest[-2:], initial[-2:])
    initialized, _ = prepare_bonds(x, neighbors, None, prm)
    assert np.array_equal(initialized, reconnected)


@pytest.mark.parametrize('relax,control,fraction', [(True, False, 0.), (False, True, 0.), (True, True, .2)])
def test_layer_geometry_translates_with_cloud_and_preserves_input(relax, control, fraction):
    # Irregular positions avoid tie-breaking ambiguity in nearest-neighbor sets.
    x = np.random.default_rng(42).uniform(-1., 1., (128, 3)).astype(np.float32)
    original = x.copy()
    cfg = PipelineConfig(device='cpu', T=20, layer_relax=relax, layer_ctrl=control,
                         layer_k=8, layer_frac=fraction, disc_ref=True, mass_ref_n=64)
    before = deepcopy(cfg)
    layer, spacing = prepare_layer_geometry(x, cfg)
    translated, spacing_shifted = prepare_layer_geometry(x+np.array([1., -2., .5], np.float32), cfg)
    assert np.array_equal(x, original) and cfg == before
    assert spacing == pytest.approx(spacing_shifted, rel=1e-6)
    assert layer[2].shape == (len(x), 16)  # Same reference mass at twice the count.
    assert np.array_equal(layer[2], translated[2])
    for index in (0, 1, 3):
        np.testing.assert_allclose(layer[index], translated[index], rtol=2e-5, atol=2e-6)
    assert layer[4] == (fraction or 1/cfg.T) if relax else layer[4] == 0.


def test_disabled_layer_requires_no_point_geometry():
    cfg = PipelineConfig(layer_relax=False, layer_ctrl=False)
    assert prepare_layer_geometry(None, cfg) == (None, None)
