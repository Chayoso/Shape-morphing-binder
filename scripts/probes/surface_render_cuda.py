"""P302 CUDA operator audit; run on hyde06, not a local simulation.

Analytic adjoints are piecewise: KNN/support/donor/tile ordering are discrete.
The finite difference below isolates packed covariance rasterization on fixed
primitives. The 300k audit checks the full live primitive path for finite gradients.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import warp as wp

wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
from physmorph.render.studio import StudioRaster
from physmorph.render.surface_gaussians import SurfacePrimitives, tangent_covariance
from physmorph.pipeline.surface_render_loss import SurfaceRenderViews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/data/relcfd/chayo/physmorph_v2')
    parser.add_argument('--out', required=True)
    parser.add_argument('--detail-height', type=int, default=2160)
    parser.add_argument('--views', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(302)
    wp.init()
    start = time.monotonic()
    device = 'cuda:0'
    center = torch.zeros(3, device=device)
    x = torch.randn(48, 3, device=device)*.4
    normals = torch.nn.functional.normalize(torch.randn_like(x), dim=1)
    sigma = x.new_full((len(x),), .08)
    opacity = x.new_full((len(x),), .25)
    direct = StudioRaster(center, 1.5, 256, 144, 35., 18., direct_covariance=True, coverage_only=True)
    old = StudioRaster(center, 1.5, 256, 144, 35., 18., coverage_only=True)
    def primitive(scale):
        return SurfacePrimitives(normals, tangent_covariance(normals, sigma*scale), opacity, sigma, opacity)
    with torch.no_grad():
        image = direct.coverage(x, primitive(1.))
        original = old.coverage(x, primitive(1.))
        parity = float((image-original).abs().max())
        assert parity < 2e-5, f'packed/eigendecomposed forward mismatch: {parity}'
    direction = torch.randn_like(x)*.15
    weight = torch.linspace(.2, 1., image.shape[1], device=device)[None]
    scalar = x.new_zeros((), requires_grad=True)
    def objective(t):
        return (direct.coverage(x+t*direction, primitive(1.+.1*t))*weight).mean()
    grad, = torch.autograd.grad(objective(scalar), scalar)
    derivatives = []
    for eps in (1e-2, 3e-3, 1e-3):
        with torch.no_grad():
            fd = (objective(eps)-objective(-eps))/(2*eps)
        rel = float((fd-grad).abs()/grad.abs().clamp_min(1e-8))
        derivatives.append(dict(eps=eps, analytic=float(grad), finite_difference=float(fd), relative=rel))
    print(json.dumps(dict(stage='packed_covariance', parity_max=parity, derivatives=derivatives)), flush=True)
    # Full-frame tails cross the rasterizer's hard alpha threshold. Isolate a
    # single primitive's core to validate the continuous backward independently.
    x = torch.tensor([[.015, .025, .01]], device=device)
    normals = torch.nn.functional.normalize(torch.tensor([[.3, .4, 1.]], device=device), dim=1)
    sigma = x.new_full((1,), .08); opacity = x.new_full((1,), .7)
    direction = x.new_tensor([[.03, -.02, .01]])
    with torch.no_grad():
        base_image = direct.coverage(x, primitive(1.))
        mask = base_image > .4
    core_derivatives = []
    for mode in ('mean', 'covariance', 'both'):
        def core_objective(t):
            p = x+t*direction if mode != 'covariance' else x
            scale = 1.+.1*t if mode != 'mean' else 1.
            return (direct.coverage(p, primitive(scale))*weight)[mask].mean()
        scalar = x.new_zeros((), requires_grad=True)
        analytical, = torch.autograd.grad(core_objective(scalar), scalar)
        for eps in (1e-2, 3e-3, 1e-3):
            with torch.no_grad():
                fd = (core_objective(eps)-core_objective(-eps))/(2*eps)
            relative = float((fd-analytical).abs()/analytical.abs().clamp_min(1e-8))
            core_derivatives.append(dict(mode=mode, eps=eps, analytic=float(analytical),
                                         finite_difference=float(fd), relative=relative))
        assert min(r['relative'] for r in core_derivatives if r['mode'] == mode) < .02, core_derivatives
    print(json.dumps(dict(stage='fixed_core_derivative', derivatives=core_derivatives)), flush=True)
    with np.load(Path(args.root)/'repro/current_pair/source_render_full_dt_iso_nn.npz') as archive:
        src = torch.tensor(archive['src'], device=device)
        target = torch.tensor(archive['tgt'], device=device)
    cameras = [(az, el) for az in np.linspace(0., 2*np.pi, 6, endpoint=False) for el in (-.3, 0., .3)]
    shared = SurfaceRenderViews(target, cameras, view_count=args.views, detail_height=args.detail_height)
    # One fixed paced reference, detached for all candidate evaluations.
    reference = src.clone()
    reference[:, 1] += shared.geometry.spacing
    window = shared.prepare(reference, 'operator_synthetic_translation')
    candidate = src.detach().clone().requires_grad_()
    value, parts = window(candidate)
    gradient, = torch.autograd.grad(value, candidate)
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
    translation = torch.zeros_like(candidate); translation[:, 1] = 1.
    slope = float((gradient*translation).sum())
    live_derivatives = []
    with torch.no_grad():
        for fraction in (.1, .03, .01):
            h = fraction*shared.geometry.spacing
            positive, _ = window(candidate.detach()+h*translation)
            negative, _ = window(candidate.detach()-h*translation)
            fd = float((positive-negative)/(2*h))
            live_derivatives.append(dict(step_sp=fraction, analytic=slope, finite_difference=fd,
                                         relative=abs(fd-slope)/max(abs(slope), 1e-8)))
    print(json.dumps(dict(stage='live_cloud_directional', derivatives=live_derivatives)), flush=True)
    repeat, _ = window(candidate.detach())
    assert abs(float(repeat-value.detach())) < 1e-6
    torch.cuda.synchronize()
    result = dict(status='core_adjoint_pass_global_fd_open', N=len(src),
                  seconds=time.monotonic()-start, packed_parity_max=parity,
                  packed_directional_derivatives=derivatives,
                  fixed_core_derivatives=core_derivatives,
                  live_cloud_derivatives=live_derivatives,
                  full_loss=float(value.detach()), components={k: float(v.detach()) for k, v in parts.items()},
                  full_gradient_norm=float(gradient.norm()),
                  full_gradient_max=float(gradient.abs().max()),
                  torch_peak_bytes=torch.cuda.max_memory_allocated(), model=window.metadata(),
                  scope='packed raster derivative + live full-cloud finite adjoint; no MPM trajectory/quality claim')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
