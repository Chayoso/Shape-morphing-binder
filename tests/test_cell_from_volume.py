"""--cell_from_volume (SHOWCASE, an option, off by default): the shape's cell from the source's volume.

By the default rule the cell is the source's bounding-box diagonal / 26, and every source is normalised to a diagonal of
8 wu, so a less compact source (a smaller volume in the same box) gets a coarser cell relative to its body: at 300k the
armadillo 742 particles a cell against the sphere's 183. With the flag the cell is the default source's (the reference
sphere, sampled alike) scaled by the cube root of the volume ratio, so every source gets the reference's particles per
cell.

(1) for the default source the flag changes nothing, bit for bit (both ratios are exactly 1.0);
(2) from another source the particles per cell are the reference's, the cell scales as (V / V_ref)^(1/3), and the
    source and target samples are unchanged;
(3) the flag is off by default (the frozen recipe) and reaches the prepare stage.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest
import torch


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA (prepare measures the discretisation on the GPU)")


def _prep(src, tgt, n, on):
    from physmorph.prepare import prepare
    return prepare(src, tgt, n, 97, 26.0, 1.4e5, 0.2, log=lambda s: None, loss_ref_n=40000, match_density=True,
                   cell_from_volume=on)


@pytest.mark.parametrize("n", [3000, 40000])
def test_the_default_source_is_unchanged_bit_for_bit(n):
    _cuda()
    off, on = _prep("assets/isosphere.obj", "assets/bunny.obj", n, False), _prep("assets/isosphere.obj", "assets/bunny.obj", n, True)
    assert dataclasses.asdict(on.prm) == dataclasses.asdict(off.prm)
    assert (on.loss_res, on.unit_ref_res, on.ppc, on.nn_berth_k, on.v_src, on.v_tgt, on.cell_shape) == (
        off.loss_res, off.unit_ref_res, off.ppc, off.nn_berth_k, off.v_src, off.v_tgt, off.cell_shape)
    assert np.array_equal(on.src, off.src) and np.array_equal(on.tgt, off.tgt)


def test_another_source_gets_the_reference_particles_per_cell():
    _cuda()
    n = 40000
    sphere = _prep("assets/isosphere.obj", "assets/homer.obj", n, False)
    off, on = _prep("assets/armadilo.obj", "assets/homer.obj", n, False), _prep("assets/armadilo.obj", "assets/homer.obj", n, True)
    assert off.ppc > 3.0 * sphere.ppc                               # the default rule: a much coarser cell for the body
    assert on.ppc == pytest.approx(sphere.ppc, rel=1e-6)           # the option: the sphere's particles per cell
    k = (on.ppc / off.ppc) ** (1.0 / 3.0)                           # dx_on / dx_off requested
    req_on = (on.v_src * on.ppc / n) ** (1.0 / 3.0)
    assert req_on == pytest.approx(k * (off.v_src * off.ppc / n) ** (1.0 / 3.0), rel=1e-9)
    assert req_on * (1.0 - 1.0 / on.prm.nx) < on.prm.dx <= req_on * (1.0 + 1e-6)   # the domain's tiling rounds it down
    assert on.prm.dx < off.prm.dx and on.loss_res > off.loss_res    # a finer cell, the loss grid following it
    assert on.cell_shape is None                                    # the option redefines the shape's cell itself
    assert np.array_equal(on.src, off.src) and np.array_equal(on.tgt, off.tgt)
    assert (on.v_src, on.v_tgt) == (off.v_src, off.v_tgt)


def test_off_by_default_and_reaches_prepare(monkeypatch):
    from scripts.pipeline_run import parse_args
    monkeypatch.setattr(sys, "argv", ["pipeline_run.py"])
    assert parse_args().cell_from_volume is False
    monkeypatch.setattr(sys, "argv", ["pipeline_run.py", "--cell_from_volume"])
    assert parse_args().cell_from_volume is True
