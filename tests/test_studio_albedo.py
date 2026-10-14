"""StudioRaster's display-only colour per primitive (albedo=..., the render gradient's videos, RGV): the deliverable
path (albedo=None) is the old one bit for bit, a uniform colour equal to the body's grey draws the same picture to the
rasterizer's rounding, and a colour changes only the covered pixels."""
import math

import pytest
import torch


def _sphere(n=4000, dev="cuda"):
    i = torch.arange(n, device=dev, dtype=torch.float32) + .5
    phi = torch.acos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** .5) * i
    nrm = torch.stack((torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)), 1)
    x = nrm.clone()
    sig = 0.06
    rr = torch.where(nrm[:, :1].abs() < .9, nrm.new_tensor((1., 0., 0.)), nrm.new_tensor((0., 1., 0.))).expand_as(nrm)
    tt = torch.nn.functional.normalize(torch.linalg.cross(nrm, rr), dim=1)
    rot = torch.stack((tt, torch.linalg.cross(nrm, tt), nrm), dim=2)
    cov = (rot * torch.tensor((sig ** 2, sig ** 2, (sig / 4) ** 2), device=dev)[None, None]) @ rot.transpose(1, 2)
    return x, nrm, cov, torch.full((n,), .92, device=dev)


def _cam():
    from physmorph.render.studio import StudioRaster
    return StudioRaster(torch.zeros(3, device="cuda"), 1.0, 192, 108, 35., 18.)


@pytest.fixture
def scene():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    pytest.importorskip("diff_gauss")
    return _cam(), _sphere()


def test_no_albedo_is_the_deliverable_path(scene):
    cam, (x, n, cov, op) = scene
    a = cam(x, n, cov, op)
    b = cam(x, n, cov, op, albedo=None)
    assert torch.equal(a, b)


def test_the_body_grey_per_primitive_draws_the_same_picture(scene):
    cam, (x, n, cov, op) = scene
    a = cam(x, n, cov, op)
    b = cam(x, n, cov, op, albedo=cam.albedo.expand(len(x), 3).contiguous())
    assert float((a - b).abs().max()) < 2.5 / 255


def test_a_colour_changes_only_the_covered_pixels(scene):
    cam, (x, n, cov, op) = scene
    a, cov_a, _ = cam(x, n, cov, op, return_buffers=True)
    red = torch.tensor((.8, .1, .05), device="cuda").expand(len(x), 3).contiguous()
    b = cam(x, n, cov, op, albedo=red)
    empty = cov_a == 0
    assert bool(empty.any()) and bool((~empty).any())
    assert torch.equal(a[empty], b[empty])                       # the backdrop is untouched
    body = cov_a > .99
    warm = lambda im: float((im[body][:, 0] - im[body][:, 2]).mean())   # noqa: E731
    assert warm(b) > warm(a) + .2                                # red where the body is
