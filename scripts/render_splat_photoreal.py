"""Render saved MPM particles as native-resolution Gaussian splats with studio lighting.

CUDA owns neighbourhoods, density normals, covariance, rasterisation and shading.
Archive decoding and image/video I/O are explicit host boundaries. Surface support,
opacity and splat radii follow render_splat_gpu.py; no mesh or hole filling is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf
import warp as wp

if os.environ.get('WARP_CACHE_PATH'):
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph.render.covariance_torch import decompose_cov_torch, world_to_view_torch
from physmorph.render.knn_gpu import knn_self_torch
from physmorph.render.settled import SettledAppearance, pin_start_frames
from physmorph.render.support import (live_support, normal_filter_size, filter_normal_buffer,
                                      MaterialShadingNormals)


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
                    weight = torch.ones(len(x), device=x.device)
                    for axis, step in enumerate(offset):
                        weight *= frac[:, axis] if step else 1 - frac[:, axis]
                    grid.index_add_(0, (index[0] * dims + index[1]) * dims + index[2], weight)
        grid = grid.reshape(1, 1, dims, dims, dims)
        for axis in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = len(self.weights)
            grid = nnf.conv3d(grid, self.weights.reshape(shape),
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
    def __init__(self, center, radius, width, height, azimuth, elevation):
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

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

    def __call__(self, x, normals, covariance, opacity, *, normal_kernel=3, return_buffers=False):
        scales, rotations = decompose_cov_torch(covariance)
        def raster(colors):
            result = self.raster(x, torch.zeros_like(x), opacity[:, None], shs=None,
                                 colors_precomp=colors.contiguous(), scales=scales,
                                 rotations=rotations)
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


def validate_pins_cuda(frames, starts, stop, device):
    """Check all raw delivered frames, not just the temporally subsampled render."""
    starts = torch.as_tensor(starts, device=device)
    anchors = torch.zeros((len(starts), 3), device=device)
    for index in range(stop):
        x = torch.as_tensor(np.asarray(frames[index], np.float32), device=device)
        new = starts == index
        anchors[new] = x[new]
        active = starts < index
        if not torch.equal(x[active], anchors[active]):
            raise ValueError(f'active pins moved at raw frame {index}; rendering refused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--azimuth', type=float, default=35)
    parser.add_argument('--elevation', type=float, default=18)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=0, help='Benchmark prefix only; zero renders all selected frames')
    parser.add_argument('--smooth-support', action='store_true',
                        help='Current 8NN compact smoothstep support; width is one target native spacing')
    parser.add_argument('--scale-normal-filter', action='store_true',
                        help='Scale the odd image-normal footprint from height1080; coverage is unchanged')
    parser.add_argument('--compare-artifacts', action='store_true',
                        help='Export baseline/support/filter/combined using exactly shared positions and attributes')
    parser.add_argument('--material-shading', action='store_true',
                        help='Guarded fixed32-neighbor affine transport of shading normals only')
    parser.add_argument('--compare-material-shading', action='store_true',
                        help='Matched baseline versus material shading; identical covariance/live opacity/filter')
    args = parser.parse_args()
    if min(args.width, args.height, args.stride, args.fps) <= 0 or args.width % 2 or args.height % 2 or args.max_frames < 0:
        parser.error('positive even dimensions, positive stride and fps are required')
    if args.compare_artifacts and (args.smooth_support or args.scale_normal_filter):
        parser.error('--compare-artifacts chooses all four settings; omit individual switches')
    if (args.material_shading or args.compare_material_shading) and (
            args.compare_artifacts or args.smooth_support or args.scale_normal_filter):
        parser.error('material shading comparison preserves baseline support and normal filter')
    if args.material_shading and args.compare_material_shading:
        parser.error('choose --material-shading or --compare-material-shading')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; there is no CPU render fallback')
    if os.environ.get('PHYSMORPH_KNN') == 'cpu':
        raise RuntimeError('CPU KNN is incompatible with this renderer')
    device = torch.device('cuda:0')
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if any(args.frames_dir.rglob('*.png')) or args.out.exists():
        raise ValueError('use an empty frames directory to avoid encoding stale frames')
    compare_mode = args.compare_artifacts or args.compare_material_shading
    variants = ([('baseline', False, False, False), ('support', True, False, False),
                 ('normal_filter', False, True, False), ('combined', True, True, False)]
                if args.compare_artifacts else
                [('baseline', False, False, False), ('material_shading', False, False, True)]
                if args.compare_material_shading else
                [('selected', args.smooth_support, args.scale_normal_filter, args.material_shading)])
    outputs, directories = {}, {}
    for name, _, _, _ in variants:
        outputs[name] = args.out.with_stem(args.out.stem + '_' + name) if compare_mode else args.out
        directories[name] = args.frames_dir / name if compare_mode else args.frames_dir
        if outputs[name].exists():
            raise ValueError(f'refusing to overwrite {outputs[name]}')
        directories[name].mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    suffix = '_render_full_dt_iso_nn.npz'
    if not str(args.npz).endswith(suffix):
        raise ValueError('requires run JSON and render_full_dt_iso_nn archive for pin validation')
    report = json.loads(Path(str(args.npz)[:-len(suffix)] + '.json').read_text())
    arm = report['arms']['render_full_dt_iso_nn']
    archive = np.load(args.npz, allow_pickle=True)
    frames = archive['frames']
    count = min(len(frames), int(archive['deliver_n'])) if 'deliver_n' in archive.files else len(frames)
    selected = sorted(set(range(0, count, args.stride)) | {count - 1})
    if args.max_frames:
        selected = selected[:args.max_frames]
    starts = pin_start_frames(np.asarray(archive['pinned'], bool), archive['pinned_at'], arm['history'], arm['config'])
    validate_pins_cuda(frames, starts, count, device)
    settled = SettledAppearance(starts, device)
    target = torch.as_tensor(np.asarray(archive['tgt'], np.float32), device=device)
    center = target.mean(0)
    radius = float((target - center).norm(dim=1).max())
    target_d, _ = knn_self_torch(target, 9)
    spacing, coverage_radius = float(target_d[:, 1].median()), float(target_d[:, 8].median())
    normals_from_density = DensityNormals(center, radius, spacing)
    studio = StudioRaster(center, radius, args.width, args.height, args.azimuth, args.elevation)
    material = MaterialShadingNormals(len(frames[0]), device) if any(v[3] for v in variants) else None
    shading_latch = SettledAppearance(starts, device) if material is not None else None
    cohort_index = min(480, count-1)
    upper_floor = float(target[:, 1].min()+.75*(target[:, 1].max()-target[:, 1].min()))
    upper = torch.as_tensor(np.asarray(frames[cohort_index], np.float32), device=device)[:, 1] >= upper_floor
    setup_seconds = time.perf_counter() - started
    timings = []
    comparison, previous = [], {}
    print(json.dumps(dict(stage='setup', seconds=setup_seconds, frames=len(selected),
                          particles=len(frames[0]), spacing=spacing)), flush=True)
    with torch.inference_mode():
        for output_index, raw_index in enumerate(selected):
            tick = time.perf_counter()
            x = torch.as_tensor(np.asarray(frames[raw_index], np.float32), device=device)
            distances, neighbors = knn_self_torch(x, 33)
            support = live_support(distances, coverage_radius, spacing)
            compact_support = live_support(distances, coverage_radius, spacing, smooth=True)
            sigma = spacing * (distances[:, 8] / coverage_radius).clamp(1., 4.)
            normals, magnitude = normals_from_density(x)
            strong = magnitude >= torch.quantile(magnitude[::max(1, len(x) // 100000)], .6)
            nearest = neighbors[:, 1:33]
            strong_neighbors = strong[nearest]
            chosen = nearest[torch.arange(len(x), device=device), strong_neighbors.float().argmax(1)]
            normals = torch.where((~strong & strong_neighbors.any(1))[:, None], normals[chosen], normals)
            for _ in range(2):
                normals = nnf.normalize(normals[neighbors].mean(1), dim=1, eps=1e-9)
            normals, sigma, support = settled.apply(raw_index, x, normals, sigma, support)
            reference = torch.where(normals[:, :1].abs() < .9,
                                    x.new_tensor((1., 0., 0.)), x.new_tensor((0., 1., 0.))).expand_as(x)
            tangent = nnf.normalize(torch.linalg.cross(normals, reference), dim=1, eps=1e-9)
            rotation = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
            variance = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
            covariance = (rotation * variance[:, None]) @ rotation.transpose(1, 2)
            material_stats = None
            if material is not None:
                shader_normals, status = material.update(x, normals, neighbors, strong, shading_latch.anchored)
                shader_normals, _, _ = shading_latch.apply(raw_index, x, shader_normals, sigma, support)
                material_stats = {region: dict(total=int(mask.sum()), **{
                    key: int((value & mask).sum()) for key, value in status.items()})
                    for region, mask in [('global', torch.ones_like(upper)), ('upper_ear', upper)]}
            current = {}
            for name, smooth_support, scaled_filter, material_shading in variants:
                selected_support = compact_support if smooth_support else support
                kernel = normal_filter_size(args.height, scaled_filter)
                image, coverage, pixel_normal = studio(
                    x, shader_normals if material_shading else normals, covariance, .92 * selected_support,
                    normal_kernel=kernel, return_buffers=True)
                current[name] = (image, coverage, pixel_normal)
                if compare_mode:
                    base = current['baseline']
                    delta = coverage - base[1]
                    row = dict(raw=raw_index, variant=name, coverage_mean=float(coverage.mean()),
                               coverage_half_fraction=float((coverage >= .5).float().mean()),
                               coverage_delta_mean=float(delta.mean()),
                               coverage_increase_max=float(delta.max().clamp_min(0)),
                               coverage_increase_gt_2e4_pixels=int((delta > 2e-4).sum()),
                               coverage_decrease_max=float((-delta).max().clamp_min(0)),
                               coverage_decrease_gt_01_pixels=int((delta < -.01).sum()),
                               baseline_half_pixels=int((base[1] >= .5).sum()),
                               baseline_half_lost_pixels=int(((base[1] >= .5) & (coverage < .5)).sum()),
                               support_mean=float(selected_support.mean()),
                               support_increase_max=float((selected_support-support).max().clamp_min(0)))
                    if material_shading:
                        row['material_shading'] = material_stats
                    if previous:
                        before = previous[name]
                        # Identical baseline-derived pixel mask for every variant; no scene-motion removal.
                        mask = (base[1] >= .5) & (previous['baseline'][1] >= .5)
                        rgb_delta = (image-before[0]).abs().mean(-1)
                        normal_delta = (pixel_normal-before[2]).norm(dim=-1)
                        row.update(previous_raw=selected[output_index-1], mask_pixels=int(mask.sum()),
                                   rgb_mae_full=float(rgb_delta.mean()),
                                   rgb_mae_baseline_mask=float(rgb_delta[mask].mean()) if mask.any() else None,
                                   normal_l2_baseline_mask=float(normal_delta[mask].mean()) if mask.any() else None,
                                   coverage_mae=float((coverage-before[1]).abs().mean()))
                    comparison.append(row)
                pixels = (image * 255 + .5).byte().cpu().numpy()
                Image.fromarray(pixels).save(directories[name] / f'{output_index:04d}.png')
            previous = current if compare_mode else {}
            elapsed = time.perf_counter() - tick
            timings.append(elapsed)
            print(json.dumps(dict(stage='frame', index=output_index, raw=raw_index,
                                  seconds=elapsed, remaining_seconds=(len(selected)-output_index-1)*float(np.mean(timings[-5:])))), flush=True)
    encode_started = time.perf_counter()
    for name, _, _, _ in variants:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                        '-framerate', str(args.fps), '-i', str(directories[name] / '%04d.png'),
                        '-c:v', 'h264_nvenc', '-gpu', '0', '-preset', 'p6', '-rc', 'vbr',
                        '-cq', '18', '-b:v', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(outputs[name])], check=True)
    metadata = dict(source=str(args.npz.resolve()), output=str(args.out.resolve()),
                    width=args.width, height=args.height, fps=args.fps, raw_frame_indices=selected,
                    view=dict(azimuth=args.azimuth, elevation=args.elevation),
                    material='uniform satin ceramic, GGX roughness0.36; synthetic studio lighting',
                    geometry='saved particles, density-normal discs; no reconstruction or hole filling',
                    support='live 8NN density support; optional compact smoothstep inside existing radius',
                    spacing_wu=spacing, coverage_radius_wu=coverage_radius,
                    support_transition_wu=spacing, support_transition_sp=1., opacity=.92,
                    sigma_rule='spacing * clamp(current r8 / target median r8, 1, 4); pin values frozen',
                    normal_filter_reference_height=1080,
                    material_shading=dict(degree=32, rest_rank_ratio=1e-4, relative_det_min=1e-4,
                                          residual_limit=.5, invalid='current refit; reanchor exposed valid graph',
                                          anchor_cadence='first exposed selected/rendered frame; no hidden raw-frame updates',
                                          upper_ear_cohort_raw=cohort_index, upper_ear_y_min=upper_floor,
                                          upper_ear_count=int(upper.sum())),
                    variants={name: dict(smooth_support=smooth, scale_normal_filter=scaled, material_shading=mat,
                                         normal_filter_size=normal_filter_size(args.height, scaled),
                                         output=str(outputs[name])) for name, smooth, scaled, mat in variants},
                    settled_freeze=True, all_raw_pin_frames_validated=count,
                    setup_seconds=setup_seconds, frame_seconds=timings,
                    encode_seconds=time.perf_counter()-encode_started, total_seconds=time.perf_counter()-started,
                    source_config=arm['config'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    support_file = Path(__file__).resolve().parents[1] / 'physmorph/render/support.py'
    metadata['support_sha256'] = hashlib.sha256(support_file.read_bytes()).hexdigest()
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    if comparison:
        args.out.with_suffix('.comparison.json').write_text(json.dumps(dict(
            definitions='Unencoded sRGB MAE and unit normal-vector L2. Same baseline alpha>=0.5 in consecutive '
                        'frames defines all variant masks. Includes real motion, coverage and shading changes; '
                        'not an isolated oscillation or physical-hole metric. Final held pair is labeled by raw indices.',
            rows=comparison), indent=2))
    print(json.dumps(dict(stage='done', outputs={name: str(path) for name, path in outputs.items()},
                          seconds=metadata['total_seconds'])), flush=True)


if __name__ == '__main__':
    main()
