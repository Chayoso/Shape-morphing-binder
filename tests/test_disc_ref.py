"""config.disc_ref (2026-09-24): the reference-discretisation factor and the counts it scales."""
import numpy as np

from physmorph.pipeline.config import PipelineConfig, disc_ref_factor


def test_factor_is_one_at_and_below_the_reference_and_when_off():
    cfg = PipelineConfig(disc_ref=True, mass_ref_n=40000)
    assert disc_ref_factor(40000, cfg) == 1.0
    assert disc_ref_factor(20000, cfg) == 1.0
    assert disc_ref_factor(300000, PipelineConfig(disc_ref=False, mass_ref_n=40000)) == 1.0
    assert disc_ref_factor(300000, PipelineConfig(disc_ref=True, mass_ref_n=0)) == 1.0


def test_factor_is_the_cube_root_of_the_particle_ratio():
    cfg = PipelineConfig(disc_ref=True, mass_ref_n=40000)
    f = disc_ref_factor(300000, cfg)
    assert abs(f - 7.5 ** (1.0 / 3.0)) < 1e-12
    assert abs(f ** 3 - 7.5) < 1e-9


def test_layer_relax_data_takes_the_asymmetry_count():
    from physmorph.render.surface_recon import layer_relax_data
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(4000, 3)).astype(np.float32)
    x = x[np.linalg.norm(x, axis=1) < 1.0]
    sp = 2.0 / 4000 ** (1.0 / 3.0)
    m8, _, nb8, _ = layer_relax_data(x, sp, k=8, k_asym=8)
    m32, _, nb32, _ = layer_relax_data(x, sp, k=8, k_asym=32)
    assert m8.sum() > 0 and m32.sum() > 0
    assert not np.array_equal(m8, m32)            # the asymmetry count changes the layer
    assert nb8.shape == nb32.shape == (len(x), 8)  # the relaxation count is the other argument
