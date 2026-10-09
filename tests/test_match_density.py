"""--match_density (D132): the source sample has the target sample's number density.

prepare() matches the meshes' volumes at a 110^3 fill, but each stratified sample is one jittered particle per voxel of
the sampler's own, coarser fill, and the two fills differ by a surface term of the shape: the 40k bunny's target sample
was 1.15 % less dense than the sphere's, which a body that conserves its volume (--volume_exact) cannot reach. With the
flag the source is rescaled about its centre so that its sample represents the target sample's fill volume.

(1) the fill volume is the sampler's own: n of its voxels hold the sample, one particle each;
(2) off: prepare() is the code as it was (no density record, the same samples);
(3) on: the target and everything on it are bit for bit the default's, the source is the default's times one scale, and
    after it the two samples represent the same volume (equal number densities).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physmorph.sampling.mesh import _stratified_fill, load_mesh, stratified_draws, stratified_fill_volume


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSMORPH_CACHE", str(tmp_path))


def test_the_fill_volume_is_the_voxels_the_sample_is_drawn_from():
    mesh = load_mesh("assets/bunny.obj")
    n = 3000
    hi, centers, pitch = _stratified_fill(mesh, n)
    assert stratified_fill_volume(mesh, n) == pytest.approx(len(centers) * pitch ** 3, rel=1e-12)
    x = stratified_draws(mesh, n, [5])[0]
    # one particle per voxel: every particle within half a pitch of a distinct fill voxel's centre
    d = np.abs(x[:, None, :] - centers[None, :, :]).max(2)
    owner = d.argmin(1)
    assert np.all(d[np.arange(n), owner] <= 0.5 * pitch + 1e-5)
    assert len(np.unique(owner)) == n
    assert len(centers) >= n


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA (prepare measures the discretisation on the GPU)")


def test_off_is_the_code_as_it_was():
    _cuda()
    from physmorph.prepare import prepare
    a = prepare("assets/isosphere.obj", "assets/bunny.obj", 3000, 7, 26.0, 1.0, 0.3, log=lambda s: None)
    b = prepare("assets/isosphere.obj", "assets/bunny.obj", 3000, 7, 26.0, 1.0, 0.3, log=lambda s: None,
                match_density=False)
    assert a.density is None and b.density is None
    assert np.array_equal(a.src, b.src) and np.array_equal(a.tgt, b.tgt) and a.v_src == b.v_src


@pytest.mark.parametrize("tgt", ["assets/bunny.obj", "assets/dragon.obj"])
def test_on_rescales_the_source_to_the_target_samples_density(tgt):
    _cuda()
    from physmorph.prepare import prepare
    off = prepare("assets/isosphere.obj", tgt, 3000, 7, 26.0, 1.0, 0.3, log=lambda s: None)
    on = prepare("assets/isosphere.obj", tgt, 3000, 7, 26.0, 1.0, 0.3, log=lambda s: None, match_density=True)
    d = on.density
    k = d["source_scale"]
    assert np.array_equal(on.tgt, off.tgt) and on.v_tgt == off.v_tgt              # the target untouched
    assert np.allclose(on.src, off.src * k, rtol=0, atol=1e-6 * float(np.abs(off.src).max()))
    assert d["source_fill"] * k ** 3 == pytest.approx(d["target_fill"], rel=1e-9)    # equal densities n / fill
    assert on.v_src == pytest.approx(off.v_src * k ** 3, rel=1e-9)
