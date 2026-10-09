"""--cell_follows_n (D137): above the reference N the MPM cell follows the particle count.

The cell was set from the source shape alone (diag / 26): 0.295 wu at every N, six pitches at 300k (183 particles a cell),
so the dragon's gaps (1.7-4.5 cells) lay inside the cubic kernel's reach of both their sides and their material tore into
beads (D137). With the flag dx = (diag / 26) x (40000 / N)^(1/3) above 40k: the particles per cell stay 40k's, the loss
grid keeps its size (one loss cell per finer MPM cell), the u gate and the thin set keep the shape's cell.

(1) at or below the reference N the flag changes nothing, bit for bit;
(2) at 300k: the cell is the shape's / (N / 40000)^(1/3) within the domain's tiling, the domain and the loss cell are the
    old ones, the shape's cell is recorded (the old dx exactly);
(3) the target keeps the u gate on the shape's cell, and a pipeline with a finer MPM cell than the shape's runs clean.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA (prepare measures the discretisation on the GPU)")


def _prep(tgt, n, cell_ref_n):
    from physmorph.prepare import prepare
    return prepare("assets/isosphere.obj", tgt, n, 97, 26.0, 1.4e5, 0.2, log=lambda s: None, loss_ref_n=40000,
                   cell_ref_n=cell_ref_n)


@pytest.mark.parametrize("n", [3000, 40000])
def test_at_or_below_the_reference_nothing_changes(n):
    _cuda()
    off, on = _prep("assets/bunny.obj", n, 0), _prep("assets/bunny.obj", n, 40000)
    assert on.cell_shape is None and off.cell_shape is None
    assert dataclasses.asdict(on.prm) == dataclasses.asdict(off.prm)
    assert (on.loss_res, on.unit_ref_res, on.ppc, on.nn_berth_k) == (off.loss_res, off.unit_ref_res, off.ppc, off.nn_berth_k)
    assert np.array_equal(on.src, off.src) and np.array_equal(on.tgt, off.tgt)


def test_at_300k_the_cell_follows_n_and_the_loss_cell_holds():
    _cuda()
    n = 300000
    off, on = _prep("assets/dragon.obj", n, 0), _prep("assets/dragon.obj", n, 40000)
    k = (n / 40000) ** (1.0 / 3.0)
    assert on.cell_shape == off.prm.dx                              # the shape's cell, exactly the old one
    assert on.prm.grid_min == off.prm.grid_min                      # the same domain
    assert on.prm.dx * on.prm.nx == pytest.approx(off.prm.dx * off.prm.nx, rel=1e-6)
    assert on.ppc == pytest.approx(off.ppc / k ** 3, rel=1e-12)
    # the requested cell is the shape's / k; the domain's exact tiling rounds it down by less than one cell in nx
    req = (off.v_src * off.ppc / n) ** (1.0 / 3.0) / k
    assert req * (1.0 - 1.0 / on.prm.nx) < on.prm.dx <= req * (1.0 + 1e-6)
    assert 20.0 < on.ppc < 26.0                                     # 40k's particles per cell
    assert on.loss_res == off.loss_res                              # the loss grid exactly as before
    assert abs(on.loss_res - on.prm.nx) <= 0.05 * on.prm.nx          # about one loss cell per MPM cell
    assert on.unit_ref_res == off.unit_ref_res
    assert np.array_equal(on.src, off.src) and np.array_equal(on.tgt, off.tgt)


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


def _cfg(**kw):
    from physmorph.pipeline import PipelineConfig
    return PipelineConfig(T=3, iters=2, animations=3, loss_res=24, render_views=2, render_elevs=(0.0, 0.5),
                          render_res=24, dt_res=32, patience=10, c2f_event=False, loss_follows_n=True, **kw)


def test_the_u_gate_keeps_the_shapes_cell(clouds):
    _cuda()
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.target import build_target
    prm = MPMParams(dx=0.5, nx=48, ny=48, nz=48, grid_min=(-12.0,) * 3)
    t_shape = build_target(torch.as_tensor(clouds[1], device="cuda"), prm, _cfg(cell_shape=1.0))
    assert t_shape.gate[2] == (24,) * 3 and t_shape.gate[1] == pytest.approx(1.0)
    t_old = build_target(torch.as_tensor(clouds[1], device="cuda"), prm, _cfg())
    assert t_old.gate[2] == (48,) * 3 and t_old.gate[1] == pytest.approx(0.5)   # the MPM cell's grid, as before


def test_a_pipeline_on_a_finer_cell_than_the_shapes_runs_clean(clouds):
    _cuda()
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import run_pipeline
    prm = MPMParams(dx=0.5, nx=48, ny=48, nz=48, grid_min=(-12.0,) * 3)
    res = run_pipeline(*clouds, prm, _cfg(cell_shape=1.0, volume_exact="carried", assim_volume=True),
                       log=lambda *_: None)
    assert all(v == 0 for v in res["guards"].values())
    assert any(h.get("frame_end") for h in res["history"])
