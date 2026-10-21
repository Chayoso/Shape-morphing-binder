"""D159 (--render_target mesh_shell): the render's target pictures drawn from the target mesh grown by the run's own
measured exterior offset (render/mesh_shell.py, target.shell_pictures). CPU: (1) the shell field is the mesh's signed
distance less c, its discs lie at distance c outside the mesh with the distance's gradient as normals, as the
exterior's lattice makes the body's discs; (2) c is the mean signed distance of the reference discs; (3) an open mesh is
closed and turned outward; (4) the mesh in the target's frame is where the frame puts the mesh's surface points; (5) the
option's validation. CUDA: (6) the default path ignores a mesh bit for bit, and the shell's pictures of a ball's sphere
are close to the samples' (one shape, one shell)."""
import numpy as np
import pytest
import torch
import trimesh

from physmorph.pipeline.config import PipelineConfig
from physmorph.render.exterior import Lattice, ZhuBridson
from physmorph.render.mesh_shell import MeshDistance, MeshShell, closed_outward, mesh_shell_discs


def _sphere(radius=5.):
    m = trimesh.creation.icosphere(subdivisions=5, radius=radius)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def _ball(radius=5., seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=torch.float64)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + .2 * (2 * torch.rand(x.shape, generator=gen, dtype=torch.float64) - 1)
    return x[x.norm(dim=1) < radius - .5]


def test_the_shell_field_is_the_signed_distance_less_c():
    V, F = _sphere()
    dist = MeshDistance(V, F)
    shell = MeshShell(dist, _ball(), 1., .6)
    q = torch.tensor([[0., 0., 0.], [7., 0., 0.], [0., 5.6, 0.], [0., 0., -3.]], dtype=torch.float64)
    f, g, s = shell(q)
    assert np.allclose(f.numpy(), [-5.6, 1.4, 0., -2.6], atol=.01)        # faceting of the icosphere below .01
    assert np.allclose(g[1:3].numpy(), [[1., 0., 0.], [0., 1., 0.]], atol=1e-3)
    assert float(g[3, 2]) < -.99                                          # inside: sd rises toward the nearest surface
    f2, g2, _ = shell(q, grad=False)
    assert g2 is None and torch.equal(f, f2) and bool((s == 1).all())


def test_the_shell_discs_stand_c_outside_the_mesh_with_radial_normals():
    V, F = _sphere()
    x = _ball()
    refs = [x.new_tensor(np.random.default_rng(1).normal(size=(4000, 3)))]
    refs[0] = refs[0] / refs[0].norm(dim=1, keepdim=True) * 5.7         # the run's exterior: 0.7 outside the sphere
    p, n, c, info = mesh_shell_discs(Lattice(torch.zeros(3, dtype=x.dtype), 30.), .4, V, F, 1., refs)
    assert abs(c - .7) < .01 and info["c_sd"] < .01 and info["watertight"]
    assert len(p) > 1000 and info["discs"] == len(p)
    r = p.norm(dim=1)
    assert float((r - 5.7).abs().max()) < .03
    assert float((n * torch.nn.functional.normalize(p, dim=1)).sum(1).min()) > .99
    # the body's discs of the same ball, on the same lattice, are made by the same rule: one per crossed cell
    pz, _, _, _ = Lattice(torch.zeros(3, dtype=x.dtype), 30.).discs(ZhuBridson(x, 1.), .4, refine=False)
    assert .5 < len(p) / len(pz) < 2.


def test_a_shell_needs_the_exterior_outside_the_mesh():
    V, F = _sphere()
    x = _ball()
    inside = [x.new_tensor([[4., 0., 0.], [0., 4.5, 0.]])]
    with pytest.raises(ValueError, match="c > 0"):
        mesh_shell_discs(Lattice(torch.zeros(3, dtype=x.dtype), 30.), .4, V, F, 1., inside)


def test_an_open_mesh_is_closed_and_turned_outward():
    V, F = _sphere()
    Vc, Fc, loops, tight = closed_outward(V, F[5:])                     # a hole of five faces
    assert loops == 1 and tight and trimesh.Trimesh(Vc, Fc).volume > 0
    Vi, Fi, loops, tight = closed_outward(V, F[:, ::-1])                 # every face inward
    assert loops == 0 and tight and trimesh.Trimesh(Vi, Fi).volume > 0


def test_the_mesh_in_the_frame_carries_the_frame_s_surface_points(tmp_path):
    from physmorph.sampling.mesh import mesh_in_frame, surface_in_frame
    V, F = _sphere(1.)
    path = tmp_path / "s.obj"
    trimesh.Trimesh(V + np.array([.3, -.2, .1]), F).export(path)
    frame = {"offset": np.array([.1, .2, -.3]), "scale": 2.5}
    Vf, Ff = mesh_in_frame(str(path), frame)
    pts, _ = surface_in_frame(str(path), frame, 2000)
    assert float(np.abs(MeshDistance(Vf, Ff).signed(pts)).max()) < 1e-4


def test_the_option_is_validated():
    assert PipelineConfig().render_target == "samples"
    with pytest.raises(ValueError, match="render_target"):
        PipelineConfig(render_target="mesh")
    with pytest.raises(ValueError, match="render_exterior"):
        PipelineConfig(render_target="mesh_shell")
    assert PipelineConfig(render_target="mesh_shell", render_exterior=True).render_target == "mesh_shell"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_target runs on the device")
def test_the_default_path_ignores_the_mesh_and_the_shell_draws_the_ball_s_sphere():
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.target import build_target
    x = (_ball(radius=10.) * .1).numpy().astype(np.float32)               # a ball of radius ~0.95, pitch 0.1
    V, F = _sphere(.95)
    prm = MPMParams(dx=.25, nx=32, ny=32, nz=32, grid_min=(-4., -4., -4.))
    kw = dict(render_views=2, render_elevs=(0.,), render_res=48, loss_res=16, dt_res=16, render_exterior=True)
    a = build_target(x, prm, PipelineConfig(**kw))
    b = build_target(x, prm, PipelineConfig(**kw), mesh=(V, F))
    assert all(torch.equal(u, v) for u, v in zip(a.ext.sils, b.ext.sils))
    assert all(torch.equal(u, v) for u, v in zip(a.ext.shade, b.ext.shade))
    s = build_target(x, prm, PipelineConfig(render_target="mesh_shell", **kw), mesh=(V, F))
    d = float(torch.stack([(u - v).square().mean() for u, v in zip(a.ext.sils, s.ext.sils)]).mean())
    assert 0. < d < 1e-2                                                  # the same shape on its own shell
    assert all(torch.equal(u, v) for u, v in zip(a.sils, s.sils))         # nothing else of the pack changes
    with pytest.raises(ValueError, match="needs the target mesh"):
        build_target(x, prm, PipelineConfig(render_target="mesh_shell", **kw))
