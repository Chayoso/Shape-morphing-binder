"""D122, surface-adaptive sampling (`--surface_density F`, sampling.mesh.stratified_draws): at F = 1 the sampler is the
one it was, bit for bit (the reference draw written out here); at F = 2 the outer band holds twice the interior's
density at the same n, with rest-volume weights of mean 1 in the ratio 1 : 2, and the sample stands in the base
sample's frame; the per-particle spacing helpers reduce to the global ones when every particle's spacing is the base."""
import numpy as np
import pytest
import torch
import trimesh

from physmorph.sampling import mesh as sm

ISO, BUNNY = "assets/isosphere.obj", "assets/bunny.obj"


def _reference_draw(mesh, n, seed):
    """The stratified sampler as it was before D122 (one jittered particle per fill voxel, the surplus dropped)."""
    ext = float(mesh.extents.max())
    lo, hi = 20, 400
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if len(sm._fill_centers(mesh, ext / mid)) < n:
            lo = mid
        else:
            hi = mid
    centers = sm._fill_centers(mesh, ext / hi)
    pitch = ext / hi
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(centers), n, replace=False) if len(centers) > n else np.arange(n)
    jitter = (rng.uniform(-0.5, 0.5, (n, 3)) * pitch).astype(np.float32)
    return (centers[keep] + jitter).astype(np.float32)


@pytest.mark.parametrize("path, n", [(ISO, 6000), (BUNNY, 20000)])
def test_surface_density_one_is_the_sampler_as_it_was(path, n):
    mesh = sm.load_mesh(path)
    x_ref = _reference_draw(mesh, n, 5)
    x1 = sm.stratified_draws(mesh, n, [5], surface_density=1.0, band_sp=2.0)[0]
    assert x1.shape == x_ref.shape and np.array_equal(x1, x_ref)
    assert np.array_equal(sm.sample_volume_stratified(mesh, n, seed=5), x_ref)


def _depth(mesh, x):
    """Each point's distance to the mesh surface (mesh units)."""
    return np.asarray(trimesh.proximity.closest_point(mesh, x.astype(np.float64))[1])


@pytest.mark.parametrize("path, n, F", [(ISO, 20000, 2.0), (BUNNY, 20000, 2.0)])
def test_surface_density_two_doubles_the_band_at_the_same_n(path, n, F):
    mesh = sm.load_mesh(path)
    rest = {}
    xF = sm.stratified_draws(mesh, n, [5], surface_density=F, band_sp=2.0, rest=rest)[0]
    x1 = rest["base"]
    assert len(xF) == n and len(x1) == n and np.isfinite(xF).all()
    w, rep = rest["w"], rest["report"]
    # the weights: two values in the ratio 1 : F, mean 1
    vals = np.unique(w)
    assert len(vals) == 2 and abs(vals[1] / vals[0] - F) < 1e-3 and abs(float(w.mean()) - 1.0) < 1e-4
    assert abs(rep["band_particle_share"] - float((w == vals[0]).mean())) < 1e-6
    assert rep["pitch_over_base"] < 1.0 and rep["band_pitch_over_base"] < rep["interior_pitch_over_base"]
    # the geometry, independent of the sampler's own bookkeeping: within the band's depth of the MESH surface the F
    # sample holds F / (F v_b + v_i) times the base sample's count (v_b the band's volume share), the interior
    # 1 / (F v_b + v_i); the fill's boundary and the mesh differ by part of a voxel, so the band is read at a margin
    rng = np.random.default_rng(0)
    pick = rng.choice(n, 4000, replace=False)
    dF, d1 = _depth(mesh, xF[pick]), _depth(mesh, x1[pick])
    depth, v_b = rep["band_depth_mesh"], rep["band_volume_share"]
    inner = 0.6 * depth                                    # well inside the band at either lattice (the fill's
                                                           #   boundary lies up to a voxel outside the mesh)
    bF, b1 = float((dF < inner).mean()), float((d1 < inner).mean())
    iF, i1 = float((dF > 1.6 * depth).mean()), float((d1 > 1.6 * depth).mean())
    assert b1 > 0.05 and i1 > 0.05
    expect_b, expect_i = F / (F * v_b + 1 - v_b), 1.0 / (F * v_b + 1 - v_b)
    assert abs(bF / b1 - expect_b) < 0.12 * expect_b
    assert abs(iF / i1 - expect_i) < 0.12 * expect_i


def test_surface_dense_sample_stands_in_the_base_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSMORPH_CACHE", str(tmp_path))
    frame1, frameF, rest = {}, {}, {}
    x1, v1 = sm.load_normalized(ISO, 6000, 3, return_volume=True, sample="stratified", frame=frame1)
    xF, vF = sm.load_normalized(ISO, 6000, 3, return_volume=True, sample="stratified", frame=frameF,
                                surface_density=2.0, band_sp=2.0, rest=rest)
    assert np.allclose(frame1["offset"], frameF["offset"]) and abs(frame1["scale"] - frameF["scale"]) < 1e-9
    assert abs(v1 - vF) < 1e-9 and len(rest["w"]) == 6000
    assert rest["report"]["band_depth"] > 0 and rest["report"]["pitch_base"] > 0
    assert np.array_equal(rest["base"], x1)                  # the base sample, as the F = 1 run has it
    # the cache returns the same sample and weights
    rest2 = {}
    xF2 = sm.load_normalized(ISO, 6000, 3, sample="stratified", surface_density=2.0, band_sp=2.0, rest=rest2)
    assert np.array_equal(xF, xF2) and np.array_equal(rest["w"], rest2["w"])
    # and the F = 1 cache key is untouched by the flag's default
    x1b = sm.load_normalized(ISO, 6000, 3, sample="stratified")
    assert np.array_equal(x1, x1b)


def test_per_particle_spacing_helpers_reduce_to_the_global_ones(device):
    from physmorph import gpu
    from physmorph.losses.volumetric import isolation_gate
    from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing
    from physmorph.thin import outer_mask
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.uniform(-1, 1, (12000, 3)).astype(np.float32), device=device)
    x = x[x.norm(dim=1) < 1.0].contiguous()
    ones = torch.ones(len(x), device=device)
    sp = layer_spacing(x)
    assert layer_spacing(x, ones) == sp
    assert gpu.median_kth_spacing(x, 8, subsample=5000, local=ones) == gpu.median_kth_spacing(x, 8, subsample=5000)
    a, b = layer_relax_data(x, sp), layer_relax_data(x, sp, local=ones)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    assert torch.allclose(a[3], b[3], atol=1e-6, rtol=1e-5)    # a scalar division is a multiplication by its inverse
    assert torch.equal(isolation_gate(x, local=ones), isolation_gate(x))
    assert torch.equal(outer_mask(x, sp, ones), outer_mask(x, sp))
    # a denser half: its particles' own spacing is the base spacing x local, and the normalised median is the base
    half = x[:, 0] > 0
    y = torch.cat([x[~half], x[half], x[half] + 0.004 * torch.randn_like(x[half])])
    local = torch.cat([ones[~half], 0.5 ** (1 / 3) * ones[half], 0.5 ** (1 / 3) * ones[half]])
    sp_mix, sp_base = layer_spacing(y), layer_spacing(y, local)
    assert sp_mix < 0.9 * sp_base and 0.9 * sp < sp_base < 1.1 * sp
