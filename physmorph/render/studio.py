"""Shared CUDA studio camera, density normals and fixed-light display model."""
import math
from functools import lru_cache

import torch
import torch.nn.functional as nnf

from .covariance_torch import decompose_cov_torch, world_to_view_torch
from .support import filter_normal_buffer


@lru_cache(maxsize=2)
def raster_identity(backend):
    """Bind the loaded binary and Python wrapper, including the isolated patch receipt."""
    import hashlib
    import importlib
    from pathlib import Path
    if backend not in ('legacy', 'continuous'):
        raise ValueError(backend)
    module = importlib.import_module('diff_gauss' if backend == 'legacy' else 'physmorph_diff_gauss')
    paths = dict(python=Path(module.__file__), binary=Path(module._C.__file__))
    if backend == 'continuous':
        paths['build_receipt'] = Path(module.__file__).parent.parent/'physmorph_build.json'
    return {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in paths.items()}


class DensityNormals:
    def __init__(self, center, radius, spacing, blur=3.0):
        self.cell = 0.5 * blur * spacing
        self.lo = center - 1.15 * radius
        self.dims = int(math.ceil(2.3 * radius / self.cell)) + 3
        self.pad = int(math.ceil(3 * blur * spacing / self.cell))
        coord = torch.arange(-self.pad, self.pad + 1, device=center.device)
        weights = torch.exp(-0.5 * (coord * self.cell / (blur * spacing)) ** 2)
        self.weights = weights / weights.sum()

    def __call__(self, x):
        relative = (x - self.lo) / self.cell
        base = relative.floor().long()
        frac = relative - base
        dims = self.dims
        grid = x.new_zeros(dims ** 3)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    offset = (dx, dy, dz)
                    index = [(base[:, axis] + step).clamp(0, dims - 1)
                             for axis, step in enumerate(offset)]
                    weight = x.new_ones(len(x))
                    for axis, step in enumerate(offset):
                        weight = weight * (frac[:, axis] if step else 1 - frac[:, axis])
                    grid.index_add_(0, (index[0] * dims + index[1]) * dims + index[2], weight)
        grid = grid.reshape(1, 1, dims, dims, dims)
        for axis in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = len(self.weights)
            grid = nnf.conv3d(grid, self.weights.to(x.dtype).reshape(shape),
                             padding=[self.pad if d == axis else 0 for d in range(3)])
        gradient = torch.stack(torch.gradient(grid[0, 0], spacing=self.cell))
        coords = 2 * relative / (dims - 1) - 1
        sample = nnf.grid_sample(gradient[None], coords[None, :, None, None].flip(-1),
                                 align_corners=True, mode='bilinear')
        normals = -sample[0, :, :, 0, 0].T
        magnitude = normals.norm(dim=1)
        return normals / magnitude.clamp_min(1e-9)[:, None], magnitude


class StudioRaster:
    """Fixed perspective camera and GGX dielectric material; alpha is never enlarged."""
    def __init__(self, center, radius, width, height, azimuth, elevation, *, direct_covariance=False,
                 coverage_only=False, raster_backend='legacy'):
        if raster_backend == 'legacy':
            from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        elif raster_backend == 'continuous':
            from physmorph_diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        else:
            raise ValueError(f'Unknown raster backend: {raster_backend}')

        self.direct_covariance = direct_covariance

        az, el = math.radians(azimuth), math.radians(elevation)
        camera = center + 3.6 * radius * center.new_tensor(
            (math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)))
        view = world_to_view_torch(camera, center)
        tan_y = math.tan(math.radians(30) / 2)
        tan_x = tan_y * width / height
        projection = center.new_zeros(4, 4)
        projection[0, 0], projection[1, 1] = 1 / tan_x, 1 / tan_y
        projection[3, 2] = 1
        projection[2, 2] = 100 / (100 - .01)
        projection[2, 3] = -100 * .01 / (100 - .01)
        self.raster = GaussianRasterizer(GaussianRasterizationSettings(
            image_height=height, image_width=width, tanfovx=tan_x, tanfovy=tan_y,
            bg=center.new_zeros(3), scale_modifier=1., viewmatrix=view.T.contiguous(),
            projmatrix=(projection @ view).T.contiguous(), sh_degree=0, campos=camera,
            prefiltered=False, debug=False))
        if raster_backend == 'continuous':
            from .stream_adapter import OrderedDefaultRaster
            self.raster = OrderedDefaultRaster(self.raster)
        if coverage_only:
            return  # The loss does not allocate full-frame lighting buffers.
        yy, xx = torch.meshgrid(
            torch.arange(height, device=center.device, dtype=torch.float32),
            torch.arange(width, device=center.device, dtype=torch.float32), indexing='ij')
        rays = ((2 * (xx + .5) / width - 1)[..., None] * tan_x * view[0, :3]
                + (1 - 2 * (yy + .5) / height)[..., None] * tan_y * view[1, :3]
                + view[2, :3])
        self.to_camera = -nnf.normalize(rays, dim=-1)
        # Smooth studio backdrop, independent of coverage; no synthetic contact shadow.
        blend = (yy / max(1, height - 1))[..., None]
        self.background = (1 - blend) * center.new_tensor((.115, .137, .165)) + blend * center.new_tensor((.215, .237, .265))
        self.albedo = center.new_tensor((.64, .69, .72))
        self.lights = []
        # Directions are in camera coordinates, then transformed to world coordinates.
        for direction, color, intensity in (
                ((-.65, .8, -1.), (1., .91, .81), 2.6),
                ((.8, .15, -.7), (.73, .85, 1.), .85),
                ((.4, .75, .8), (.85, .92, 1.), 1.35)):
            light = nnf.normalize(center.new_tensor(direction) @ view[:3, :3], dim=0)
            self.lights.append((light, center.new_tensor(color) * intensity))

    def raster_colors(self, x, normals, covariance, opacity, colors):
        if self.direct_covariance:
            from .surface_gaussians import pack_covariance
            result = self.raster(means3D=x.contiguous(), means2D=torch.zeros_like(x),
                                 opacities=opacity[:, None].contiguous(), shs=None,
                                 colors_precomp=colors.contiguous(),
                                 cov3Ds_precomp=pack_covariance(covariance),
                                 norm3Ds_precomp=normals.contiguous())
        else:
            scales, rotations = decompose_cov_torch(covariance)
            result = self.raster(x, torch.zeros_like(x), opacity[:, None], shs=None,
                                 colors_precomp=colors.contiguous(), scales=scales, rotations=rotations)
        image = result[0] if isinstance(result, (tuple, list)) else result
        return image.clamp(0, 1).permute(1, 2, 0).flip(0).contiguous()

    def coverage(self, x, primitives):
        return self.raster_colors(x, primitives.normals, primitives.covariance,
                                  primitives.opacity, torch.ones_like(x))[..., 0]

    def __call__(self, x, normals, covariance, opacity, *, normal_kernel=3, return_buffers=False):
        if not self.direct_covariance:
            scales, rotations = decompose_cov_torch(covariance)
        def raster(colors):
            if self.direct_covariance:
                return self.raster_colors(x, normals, covariance, opacity, colors)
            result = self.raster(x, torch.zeros_like(x), opacity[:, None], shs=None,
                                 colors_precomp=colors.contiguous(), scales=scales, rotations=rotations)
            image = result[0] if isinstance(result, (tuple, list)) else result
            return image.clamp(0, 1).permute(1, 2, 0).flip(0).contiguous()
        normal_buffer = raster(.5 * (normals + 1))
        coverage = raster(torch.ones_like(normals))[..., 0]
        # Smooth only the normal estimate. Compositing uses the original coverage.
        normal = filter_normal_buffer(normal_buffer, coverage, normal_kernel)
        view = self.to_camera
        nv = (normal * view).sum(-1).clamp_min(1e-4)
        ambient = .20 + .12 * (.5 + .5 * normal[..., 1])
        radiance = self.albedo * ambient[..., None]
        # GGX with Schlick Fresnel and Smith visibility, fixed satin-ceramic roughness.
        roughness = .36
        alpha2 = roughness ** 4
        k = (roughness + 1) ** 2 / 8
        visibility_v = nv / (nv * (1 - k) + k)
        for light, energy in self.lights:
            half_vector = nnf.normalize(view + light, dim=-1)
            nl = (normal * light).sum(-1).clamp(0, 1)
            nh = (normal * half_vector).sum(-1).clamp(0, 1)
            vh = (view * half_vector).sum(-1).clamp(0, 1)
            distribution = alpha2 / (math.pi * ((nh * nh * (alpha2 - 1) + 1) ** 2).clamp_min(1e-8))
            fresnel = .04 + .96 * (1 - vh) ** 5
            visibility_l = nl / (nl * (1 - k) + k)
            specular = distribution * fresnel * visibility_v * visibility_l / (4 * nv * nl).clamp_min(1e-5)
            diffuse = self.albedo * (1 - fresnel[..., None]) / math.pi
            radiance = radiance + (diffuse + specular[..., None]) * energy * nl[..., None]
        coverage = coverage[..., None]
        linear = radiance * coverage + self.background * (1 - coverage)
        # Fixed photographic shoulder then sRGB transfer, with no frame-wise exposure fitting.
        mapped = linear / (1 + linear)
        srgb = torch.where(mapped <= .0031308, 12.92 * mapped, 1.055 * mapped.pow(1 / 2.4) - .055)
        image = srgb.clamp(0, 1)
        return (image, coverage[..., 0], normal) if return_buffers else image

