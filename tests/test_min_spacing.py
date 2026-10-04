"""The minimum spacing of the position update (kernels.k_update, D70), warp CPU: two particles pressed together in a
slab at rest are moved apart to the spacing over one window, each by half; particles that are farther apart than the
spacing do not move; without the spacing nothing moves. A particle of the outermost layer is not moved along its
normal (D71): pressed from below, it stays in its surface and the particle under it gives way."""
import numpy as np
import torch

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.window.layer import layer_relax_data

DEV = "cpu"


def _slab(n_side=8, layers=4, spacing=0.25, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(g, np.arange(layers) * spacing, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x += rng.uniform(-0.05, 0.05, x.shape).astype(np.float32) * spacing
    return x - x.mean(0), spacing


def _run(x, spacing, layer=None):
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.0, smoothing=0.955, grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)
    T = 40
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=compute_rest_volumes(x, 1.0, prm, DEV),
                    control_steps=T // 2, spacing=spacing, layer=layer)   # no elasticity, at rest: the position update alone
    tr.rollout()
    return tr.x[T].numpy()


def test_pressed_particles_are_moved_apart_and_the_rest_stays():
    x, sp = _slab()
    r = 0.9 * sp
    i = int(np.argmin(np.linalg.norm(x, axis=1)))                  # a particle inside the slab
    d = np.linalg.norm(x - x[i], axis=1)
    j = int(np.argsort(d)[1])
    x[j] = x[i] + 0.2 * r * (x[j] - x[i]) / d[j]                   # pressed to a fifth of the spacing
    nbr = torch.cdist(torch.as_tensor(x), torch.as_tensor(x)).argsort(1)[:, 1:17]
    still = _run(x.copy(), None)
    assert np.abs(still - x).max() < 1e-6                           # at rest nothing else moves a particle
    out = _run(x.copy(), (nbr, r))
    gap = np.linalg.norm(out[i] - out[j])
    assert 0.8 * r < gap <= r + 1e-5                                # 1 - (1 - 1/20)^40 = 0.87 of the overlap
    mid = 0.5 * (x[i] + x[j])
    assert np.linalg.norm(0.5 * (out[i] + out[j]) - mid) < 0.05 * r   # each by half: the pair's middle stays
    far = np.linalg.norm(x - mid, axis=1) > 2.0 * sp
    assert np.abs(out[far] - x[far]).max() < 1e-6


def test_a_layer_particle_is_not_moved_along_its_normal():
    x, sp = _slab()
    r = 0.9 * sp
    mask, nrm, nbr, w = (t.numpy() for t in layer_relax_data(torch.as_tensor(x), sp, k=8, h_sp=2.0))
    top = np.where((mask > 0.5) & (nrm[:, 1] > 0.9))[0]
    i = int(top[np.argmin(np.linalg.norm(x[top][:, [0, 2]], axis=1))])   # a layer particle in the middle of the top face
    under = np.where(mask < 0.5)[0]
    j = int(under[np.argmin(np.linalg.norm(x[under] - x[i], axis=1))])
    x[j] = x[i] - 0.2 * r * nrm[i]                                 # pressed against it from below, along its normal
    knn = torch.cdist(torch.as_tensor(x), torch.as_tensor(x)).argsort(1)[:, 1:17]
    out = _run(x.copy(), (knn, r), layer=(mask, nrm, nbr, w, 0.0))   # no relaxation: the spacing alone
    assert abs(float((out[i] - x[i]) @ nrm[i])) < 1e-5 * r          # the layer particle stays in its surface
    assert float((x[j] - out[j]) @ nrm[i]) > 0.25 * r               # the one under it gives way: 0.8 r (1 - (1 - 1/40)^40)
    free = _run(x.copy(), (knn, r))
    assert float((free[i] - x[i]) @ nrm[i]) > 0.25 * r              # without a layer it is carried outward
