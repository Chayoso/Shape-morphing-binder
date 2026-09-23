"""The exterior test of the outer layer (surface_recon.exterior_surfels; docs/surface_gradient.md 14):
surfels of a true face keep, surfels at a density step INSIDE the material are dropped."""
import numpy as np

from physmorph.render.surface_recon import exterior_surfels


def _cloud(seed=0):
    """A slab 12 x 12 x 8 spacings: the upper half at full density, the lower half at 40 % (an
    interior density step at y = 0), jittered."""
    rng = np.random.default_rng(seed)
    sp = 1.0
    pts = []
    for y in np.arange(-4, 4, sp):
        keep = 1.0 if y >= 0 else 0.4
        for xx in np.arange(-6, 6, sp):
            for zz in np.arange(-6, 6, sp):
                if rng.uniform() < keep:
                    pts.append([xx, y, zz])
    x = np.asarray(pts, np.float64) + rng.uniform(-0.15, 0.15, (len(pts), 3))
    from scipy.spatial import cKDTree
    sp8 = float(np.median(cKDTree(x).query(x, k=9, workers=-1)[0][:, -1]))   # the pipeline's 8-NN spacing
    return x.astype(np.float32), sp8


def test_true_face_kept_interior_step_dropped():
    x, sp = _cloud()
    top = x[np.abs(x[:, 1] - 3.0) < 0.3]                    # the top face particles (y ~ 3)
    step = x[np.abs(x[:, 1] - 0.0) < 0.3]                   # particles at the density step (y ~ 0)
    n_up = np.tile([0.0, 1.0, 0.0], (len(top), 1)).astype(np.float32)
    n_step = np.tile([0.0, -1.0, 0.0], (len(step), 1)).astype(np.float32)   # the step's "outward" = toward the sparse side
    p1, n1, d1 = exterior_surfels(top, n_up, x, sp)
    assert d1 <= 0.05 * len(top), d1                        # the top face is exterior: (almost) nothing dropped
    p2, n2, d2 = exterior_surfels(step, n_step, x, sp)
    assert d2 >= 0.9 * len(step), (d2, len(step))          # the step has material on its outward side: dropped
    # the sparse side's own face at the bottom (y ~ -4) is exterior too
    bot = x[np.abs(x[:, 1] + 4.0) < 0.3]
    p3, n3, d3 = exterior_surfels(bot, np.tile([0.0, -1.0, 0.0], (len(bot), 1)).astype(np.float32), x, sp)
    assert d3 <= 0.05 * max(len(bot), 1), d3


def test_fill_pockets_closes_sub_voxel_pockets_only():
    """sampling.mesh._fill_pockets: a 1-voxel pocket and a 1-voxel tunnel are filled, a 2-voxel slot
    and the outside are not."""
    from physmorph.sampling.mesh import _fill_pockets
    M = np.zeros((12, 12, 12), bool)
    M[2:10, 2:10, 2:10] = True                    # a solid block
    M[5, 5, 5] = False                            # a 1-voxel pocket
    M[2:10, 7, 7] = False                         # a 1-voxel tunnel through the block along x
    M[2:10, 3:5, 3] = False                       # a 2-voxel-wide slot (y 3..4) along x at z = 3
    out, n, it = _fill_pockets(M)
    assert out[5, 5, 5] and out[2:10, 7, 7].all(), "pocket and tunnel filled"
    assert not out[2:10, 3:5, 3].any(), "the 2-voxel slot stays open"
    assert not out[0].any() and not out[:, 0].any(), "nothing added outside"
    assert n == 1 + 8 and it >= 1, (n, it)
