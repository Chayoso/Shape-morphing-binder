"""Independent dense raster oracle and cutoff discriminator (CUDA on hyde06).

The oracle is deliberately not used by training, rendering, or physical metrics.
It has no tiles and no transmittance early termination. Pixel centers and the
screen covariance floor match the loaded diff_gauss operator under test.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

TAU = 1. / 255.


def dense_alpha(x, covariance, opacity, settings, *, mode='hard', fixed_mask=None):
    """Return [N,H,W] alpha, pre-cutoff alpha and live support mask."""
    h, w = settings.image_height, settings.image_width
    view = settings.viewmatrix.T.to(x)
    projection = settings.projmatrix.T.to(x)
    homogeneous = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    camera = homogeneous @ view.T
    projected = homogeneous @ projection.T
    ndc = projected[:, :2] / (projected[:, 3:] + 1e-7)
    pixel = ((ndc + 1) * x.new_tensor((w, h)) - 1) / 2
    z = camera[:, 2]
    ratio_x = (camera[:, 0] / z).clamp(-1.3*settings.tanfovx, 1.3*settings.tanfovx)
    ratio_y = (camera[:, 1] / z).clamp(-1.3*settings.tanfovy, 1.3*settings.tanfovy)
    fx, fy = w/(2*settings.tanfovx), h/(2*settings.tanfovy)
    zero = torch.zeros_like(z)
    jacobian = torch.stack((fx/z, zero, -fx*ratio_x/z,
                            zero, fy/z, -fy*ratio_y/z), dim=1).reshape(-1, 2, 3)
    transform = jacobian @ view[:3, :3]
    screen_cov = transform @ covariance @ transform.transpose(1, 2)
    screen_cov = screen_cov + .3*torch.eye(2, dtype=x.dtype, device=x.device)
    conic = torch.linalg.inv(screen_cov)
    yy, xx = torch.meshgrid(torch.arange(h, dtype=x.dtype, device=x.device),
                            torch.arange(w, dtype=x.dtype, device=x.device), indexing='ij')
    delta = pixel[:, None, None] - torch.stack((xx, yy), dim=-1)
    power = -.5*torch.einsum('nhwi,nij,nhwj->nhw', delta, conic, delta)
    raw = opacity[:, None, None]*power.exp()
    live = raw >= TAU
    if mode == 'hard':
        alpha = torch.where(live if fixed_mask is None else fixed_mask, raw.clamp_max(.99), 0.)
    elif mode == 'shifted':
        alpha = (raw-TAU).clamp(0., .99)
    else:
        raise ValueError(mode)
    return alpha, raw, live


def dense_coverage(x, covariance, opacity, settings, **kwargs):
    alpha, raw, live = dense_alpha(x, covariance, opacity, settings, **kwargs)
    return (1-(1-alpha).prod(dim=0)).flip(0), raw, live


def dense_rgb(x, covariance, opacity, colors, settings, mode):
    alpha, _, _ = dense_alpha(x, covariance, opacity, settings, mode=mode)
    view = settings.viewmatrix.T.to(x)
    depths = x @ view[2, :3] + view[2, 3]
    order = depths.argsort(stable=True)
    alpha = alpha[order]
    before = torch.cat((torch.ones_like(alpha[:1]), (1-alpha).cumprod(0)[:-1]))
    return ((alpha*before)[..., None]*colors[order, None, None]).sum(0).flip(0)


def boundary_and_stack_cases(raster, backend):
    settings = raster.raster_settings if hasattr(raster, 'raster_settings') else raster.raster.raster_settings
    view = settings.viewmatrix.T
    center = settings.campos + 5.4*view[2, :3]
    rows = []
    cases = [(f'tile_{px}', 1, px, .92, .14) for px in (15.25, 15.75, 16.01, 31.99, 32.01)]
    cases += [(f'stack_{n}', n, 127.5, .92, .14) for n in (8, 64, 256)]
    cases += [('saturated', 1, 127.5, 1.5, .14), ('clipped_fov', 1, 306.7, .92, 1.2)]
    for name, count, px, opacity_value, sigma in cases:
        x = center[None].repeat(count, 1)
        x += ((px-127.5)*5.4/(256/(2*settings.tanfovx)))*view[0, :3]
        x += torch.arange(count, device=x.device)[:, None]*.001*view[2, :3]
        cov = torch.eye(3, device=x.device)[None].repeat(count, 1, 1)*sigma**2
        opacity = x.new_full((count,), opacity_value)
        normals = view[2, :3][None].repeat(count, 1)
        colors = (.2+.6*torch.arange(count, device=x.device).remainder(3)/2)[:, None].repeat(1, 3)
        colors[:, 1] = .45
        weight = torch.linspace(.2, 1., 256, device=x.device)[None, :, None]
        record = dict(case=name, derivatives=[])
        with torch.no_grad():
            actual = raster.raster_colors(x, normals, cov, opacity, colors)
            expected = dense_rgb(x.double(), cov.double(), opacity.double(), colors.double(), settings,
                                 'hard' if backend == 'legacy' else 'shifted')
            record['forward_max_error'] = float((actual-expected).abs().max())
        for variable in ('mean_depth', 'scale', 'opacity', 'color'):
            def objective(t, oracle):
                points, covariance, alpha, color = [v.double() if oracle else v for v in (x, cov, opacity, colors)]
                if variable == 'mean_depth':
                    points = points+t*.05*view[2, :3].to(points)
                elif variable == 'scale':
                    covariance = covariance*(1+.1*t)**2
                elif variable == 'opacity':
                    alpha = alpha+.1*t
                else:
                    color = color+.1*t
                if oracle:
                    image = dense_rgb(points, covariance, alpha, color, settings,
                                      'hard' if backend == 'legacy' else 'shifted')
                else:
                    image = raster.raster_colors(points, normals, covariance, alpha, color)
                return (image*weight.to(image)).mean()
            t = x.new_zeros((), requires_grad=True)
            actual_grad, = torch.autograd.grad(objective(t, False), t)
            td = x.new_zeros((), dtype=torch.float64, requires_grad=True)
            expected_grad, = torch.autograd.grad(objective(td, True), td)
            with torch.no_grad():
                fd = (objective(.001, False)-objective(-.001, False))/.002
            record['derivatives'].append(dict(variable=variable, cuda=float(actual_grad),
                dense=float(expected_grad), fd=float(fd), absolute_error=float((actual_grad-expected_grad).abs()),
                relative_error=float((actual_grad-expected_grad).abs()/expected_grad.abs().clamp_min(1e-10))))
        rows.append(record)
        print(json.dumps(record), flush=True)
    if backend == 'continuous':
        def stream_eval():
            live_x = x.clone().requires_grad_()
            image = raster.raster_colors(live_x, normals, cov, opacity, colors)
            gradient, = torch.autograd.grad((image*weight).mean(), live_x)
            # Exercise temporary-storage reuse after the extension has enqueued work.
            scratch = torch.empty((1024, 1024), device=x.device).fill_(123.)
            return image.detach(), gradient, scratch
        default_image, default_grad, _ = stream_eval()
        caller = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            side_image, side_grad, _ = stream_eval()
        caller.wait_stream(side)
        rows.append(dict(case='side_stream', forward_max_error=float((default_image-side_image).abs().max()),
                         gradient_max_error=float((default_grad-side_grad).abs().max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--backend', choices=('legacy', 'continuous'), default='legacy')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    from physmorph.render.studio import StudioRaster
    from physmorph.render.surface_gaussians import SurfacePrimitives, tangent_covariance
    device = 'cuda:0'
    center = torch.zeros(3, device=device)
    options = {} if args.backend == 'legacy' else {'raster_backend': args.backend}
    raster = StudioRaster(center, 1.5, 256, 144, 35., 18., direct_covariance=True,
                          coverage_only=True, **options)
    settings = raster.raster.raster_settings
    package = importlib.import_module('diff_gauss' if args.backend == 'legacy' else 'physmorph_diff_gauss')
    bindings = {name: Path(importlib.import_module(name).__file__) for name in (
        'physmorph.render.studio', 'physmorph.render.surface_gaussians', 'physmorph.render.covariance_torch')}
    if args.backend == 'continuous':
        bindings['stream_adapter'] = Path(importlib.import_module('physmorph.render.stream_adapter').__file__)
    bindings.update(raster_python=Path(package.__file__), raster_binary=Path(package._C.__file__),
                    probe=Path(__file__))
    receipt = Path(package.__file__).parent.parent/'physmorph_build.json'
    if args.backend == 'continuous':
        bindings['build_receipt'] = receipt
    identities = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                  for name, path in bindings.items()}
    results = []
    for population in ('single', 'p302_48'):
        torch.manual_seed(302)
        if population == 'single':
            x = torch.tensor([[.015, .025, .01]], device=device)
            normals = torch.nn.functional.normalize(x.new_tensor([[.3, .4, 1.]]), dim=1)
            direction = x.new_tensor([[.03, -.02, .01]])
        else:
            x = torch.randn(48, 3, device=device)*.4
            normals = torch.nn.functional.normalize(torch.randn_like(x), dim=1)
            direction = torch.randn_like(x)*.15
        sigma = x.new_full((len(x),), .08)
        opacity = x.new_full((len(x),), .25)
        base_cov = tangent_covariance(normals, sigma)
        _, _, baseline_mask = dense_coverage(x.double(), base_cov.double(), opacity.double(), settings)
        for motion in ('mean', 'scale', 'joint'):
            def parameters(t, double=False):
                points = x.double() if double else x
                cov = base_cov.double() if double else base_cov
                if motion != 'scale':
                    points = points+t*direction.to(points)
                if motion != 'mean':
                    cov = cov*(1.+.1*t)**2
                return points, cov
            for weights in ('uniform', 'ramp'):
                weight = torch.ones((144, 256), device=device)
                if weights == 'ramp':
                    weight = weight*torch.linspace(.2, 1., 256, device=device)[None]
                images = {}
                for method in ('cuda', 'dense_hard', 'dense_frozen', 'dense_shifted'):
                    def objective(t, return_mask=False):
                        points, cov = parameters(t, double=method != 'cuda')
                        if method == 'cuda':
                            primitives = SurfacePrimitives(normals, cov, opacity, sigma, opacity)
                            image = raster.coverage(points, primitives)
                            mask = None
                        else:
                            image, _, mask = dense_coverage(points, cov, opacity.to(points), settings,
                                mode='shifted' if method == 'dense_shifted' else 'hard',
                                fixed_mask=baseline_mask if method == 'dense_frozen' else None)
                        value = (image*weight.to(image)).mean()
                        return (value, mask) if return_mask else value
                    scalar = x.new_zeros((), dtype=torch.float32 if method == 'cuda' else torch.float64,
                                         requires_grad=True)
                    value = objective(scalar)
                    gradient, = torch.autograd.grad(value, scalar)
                    rows = []
                    with torch.no_grad():
                        for step in (1e-2, 3e-3, 1e-3, 3e-4):
                            plus, mp = objective(step, True)
                            minus, mm = objective(-step, True)
                            fd = (plus-minus)/(2*step)
                            rows.append(dict(step=step, fd=float(fd), analytic=float(gradient),
                                relative=float((fd-gradient).abs()/gradient.abs().clamp_min(1e-12)),
                                raw_threshold_mask_flips=None if mp is None else int((mp != mm).sum()),
                                applied_mask_flips=0 if method == 'dense_frozen' else
                                    None if mp is None else int((mp != mm).sum())))
                    images[method] = dict(value=float(value), derivatives=rows)
                record = dict(population=population, motion=motion, weights=weights, methods=images)
                results.append(record)
                print(json.dumps(record), flush=True)
        with torch.no_grad():
            image = raster.coverage(x, SurfacePrimitives(normals, base_cov, opacity, sigma, opacity))
            oracle, _, _ = dense_coverage(x.double(), base_cov.double(), opacity.double(), settings,
                                          mode='hard' if args.backend == 'legacy' else 'shifted')
            parity = float((image-oracle).abs().max())
        results.append(dict(population=population, forward_dense_max=parity))
    stress_cases = boundary_and_stack_cases(raster, args.backend)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != identities[name]['sha256']
           for name, path in bindings.items()):
        raise ValueError('Loaded operator files changed during audit')
    result = dict(backend=args.backend, scope='in-frustum operator only; no physical-quality claim',
                  loaded_files=identities, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results,
                  boundary_and_stack=stress_cases)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
