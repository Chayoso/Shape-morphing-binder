"""Render saved MPM particles as native-resolution Gaussian splats with studio lighting.

CUDA owns neighbourhoods, density normals, covariance, rasterisation and shading.
Archive decoding and image/video I/O are explicit host boundaries. Surface support and
opacity follow render_splat_gpu.py; no mesh or hole filling is used. A disc's radius is the
target spacing times the particle's spacing on the surface (the 8th-neighbour distance in its
tangent plane) over the same on the target's surface, clamped to 1-4 (D35).
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
                                      MaterialShadingNormals, surface_particles, surface_spacing)


from physmorph.render.studio import DensityNormals, StudioRaster


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
    parser.add_argument('--compare-normal-thickness', action='store_true',
                        help='Experimental matched baseline versus reference-spacing normal thickness; tangent radii and live opacity identical')
    parser.add_argument('--surface-common', action='store_true',
                        help='P302 stateless primitives shared with surface_gs_loss; no appearance latch')
    parser.add_argument('--raster-backend', choices=('legacy', 'continuous'), default='legacy')
    args = parser.parse_args()
    if args.raster_backend == 'continuous' and not args.surface_common:
        parser.error('--raster-backend continuous requires --surface-common')
    if args.surface_common and any((args.compare_artifacts, args.smooth_support,
            args.scale_normal_filter, args.material_shading, args.compare_material_shading,
            args.compare_normal_thickness)):
        parser.error('--surface-common excludes historical appearance variants')
    if args.compare_normal_thickness and any((args.compare_artifacts, args.smooth_support,
            args.scale_normal_filter, args.material_shading, args.compare_material_shading)):
        parser.error('--compare-normal-thickness preserves baseline shading, support and image filter')
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
    compare_mode = args.compare_artifacts or args.compare_material_shading or args.compare_normal_thickness
    variants = ([('baseline', False, False, False), ('support', True, False, False),
                 ('normal_filter', False, True, False), ('combined', True, True, False)]
                if args.compare_artifacts else
                [('baseline', False, False, False), ('material_shading', False, False, True)]
                if args.compare_material_shading else
                [('baseline', False, False, False), ('reference_thickness', False, False, False)]
                if args.compare_normal_thickness else
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
    if 'pinned' in archive.files:
        starts = pin_start_frames(np.asarray(archive['pinned'], bool), archive['pinned_at'], arm['history'], arm['config'])
    else:
        # settled-transport archives carry no pins: no particle ever latches its appearance, every
        # frame is drawn from its live state (the same as a pinned archive before any admission)
        starts = np.full(len(frames[0]), np.iinfo(np.int64).max, np.int64)
    validate_pins_cuda(frames, starts, count, device)
    settled = SettledAppearance(starts, device)
    target = torch.as_tensor(np.asarray(archive['tgt'], np.float32), device=device)
    center = target.mean(0)
    radius = float((target - center).norm(dim=1).max())
    target_d, target_neighbors = knn_self_torch(target, 33)
    spacing, coverage_radius = float(target_d[:, 1].median()), float(target_d[:, 8].median())
    normals_from_density = DensityNormals(center, radius, spacing)

    def density_normals(x, neighbors):
        """(normals, strong): the density gradient, a strong neighbour's where it is weak, averaged twice."""
        normals, magnitude = normals_from_density(x)
        strong = magnitude >= torch.quantile(magnitude[::max(1, len(x) // 100000)], .6)
        nearest = neighbors[:, 1:33]
        strong_neighbors = strong[nearest]
        chosen = nearest[torch.arange(len(x), device=device), strong_neighbors.float().argmax(1)]
        normals = torch.where((~strong & strong_neighbors.any(1))[:, None], normals[chosen], normals)
        for _ in range(2):
            normals = nnf.normalize(normals[neighbors].mean(1), dim=1, eps=1e-9)
        return normals, strong

    # the disc's reference: the tangent-plane spacing of the target's own surface (D35)
    with torch.inference_mode():
        on_surface = surface_particles(target, target_neighbors, coverage_radius)
        surface_reference = float(surface_spacing(
            target, target_neighbors, density_normals(target, target_neighbors)[0])[on_surface].median())
    common = None
    if args.surface_common:
        from physmorph.render.surface_gaussians import SurfaceGaussians
        common = SurfaceGaussians(center, radius, spacing, coverage_radius)
    studio = StudioRaster(center, radius, args.width, args.height, args.azimuth, args.elevation,
                          direct_covariance=args.surface_common, raster_backend=args.raster_backend)
    material = MaterialShadingNormals(len(frames[0]), device) if any(v[3] for v in variants) else None
    shading_latch = SettledAppearance(starts, device) if material is not None else None
    cohort_index = min(480, count-1)
    upper_floor = float(target[:, 1].min()+.75*(target[:, 1].max()-target[:, 1].min()))
    upper = torch.as_tensor(np.asarray(frames[cohort_index], np.float32), device=device)[:, 1] >= upper_floor
    setup_seconds = time.perf_counter() - started
    timings = []
    comparison, previous, footprints = [], {}, []
    print(json.dumps(dict(stage='setup', seconds=setup_seconds, frames=len(selected),
                          particles=len(frames[0]), spacing=spacing)), flush=True)
    with torch.inference_mode():
        for output_index, raw_index in enumerate(selected):
            tick = time.perf_counter()
            x = torch.as_tensor(np.asarray(frames[raw_index], np.float32), device=device)
            if common is not None:
                primitive = common(x)
                normals, sigma, support = primitive.normals, primitive.sigma, primitive.support
                covariance = primitive.covariance
                compact_support = support
            else:
                distances, neighbors = knn_self_torch(x, 33)
                support = live_support(distances, coverage_radius, spacing)
                compact_support = live_support(distances, coverage_radius, spacing, smooth=True)
                normals, strong = density_normals(x, neighbors)
                sigma = spacing * (surface_spacing(x, neighbors, normals) / surface_reference).clamp(1., 4.)
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
                selected_covariance = covariance
                if name == 'reference_thickness':
                    from physmorph.render.footprint_policy import reference_thickness_covariance
                    selected_covariance = reference_thickness_covariance(rotation, sigma, spacing)
                if args.compare_normal_thickness:
                    from physmorph.render.footprint_diagnostics import summarize_world_footprints
                    observation = summarize_world_footprints(
                        selected_covariance, normals, support=selected_support, opacity=.92 * selected_support,
                        populations={'active_pinned': settled.anchored, 'free': ~settled.anchored,
                                     'upper_cohort': upper, 'sparse_support': selected_support < 1.})
                    footprints.append(dict(raw=raw_index, variant=name, observation=observation))
                    if (observation['populations']['positive_support_and_opacity']['invalid_count']
                            or observation['validation_counts']['invalid_support']
                            or observation['validation_counts']['invalid_opacity']):
                        args.out.with_suffix('.footprints.failed.json').write_text(json.dumps(
                            dict(reason='Invalid renderable footprint; comparison refused before raster', rows=footprints),
                            indent=2, allow_nan=False))
                        raise ValueError(f'Invalid positive-opacity footprint at raw frame {raw_index}, variant {name}')
                image, coverage, pixel_normal = studio(
                    x, shader_normals if material_shading else normals, selected_covariance, .92 * selected_support,
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
                    sigma_rule='spacing * clamp(tangent-plane 8th-neighbour distance / its median on the target surface, '
                               '1, 4); pin values frozen',
                    surface_reference_wu=surface_reference,
                    normal_filter_reference_height=1080,
                    material_shading=dict(degree=32, rest_rank_ratio=1e-4, relative_det_min=1e-4,
                                          residual_limit=.5, invalid='current refit; reanchor exposed valid graph',
                                          anchor_cadence='first exposed selected/rendered frame; no hidden raw-frame updates',
                                          upper_ear_cohort_raw=cohort_index, upper_ear_y_min=upper_floor,
                                          upper_ear_count=int(upper.sum())),
                    variants={name: dict(smooth_support=smooth, scale_normal_filter=scaled, material_shading=mat,
                                         normal_thickness=('reference spacing / 4' if name == 'reference_thickness'
                                                           else 'adaptive tangent sigma / 4'),
                                         normal_filter_size=normal_filter_size(args.height, scaled),
                                         output=str(outputs[name])) for name, smooth, scaled, mat in variants},
                    settled_freeze=not args.surface_common, all_raw_pin_frames_validated=count,
                    setup_seconds=setup_seconds, frame_seconds=timings,
                    encode_seconds=time.perf_counter()-encode_started, total_seconds=time.perf_counter()-started,
                    source_config=arm['config'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if common is not None:
        metadata['surface_render_model'] = common.metadata()
        from physmorph.render.studio import raster_identity
        metadata['raster_backend'] = args.raster_backend
        metadata['raster_files'] = raster_identity(args.raster_backend)
        metadata['sigma_rule'] = 'spacing * clamp(current r8 / target median r8, 1, 4); stateless'
    support_file = Path(__file__).resolve().parents[1] / 'physmorph/render/support.py'
    metadata['support_sha256'] = hashlib.sha256(support_file.read_bytes()).hexdigest()
    metadata['render_source_sha256'] = {name: hashlib.sha256(
        (support_file.parent/name).read_bytes()).hexdigest()
        for name in ('studio.py', 'surface_gaussians.py', 'knn_gpu.py', 'covariance_torch.py')}
    if args.compare_normal_thickness:
        from physmorph.render.studio import raster_identity
        metadata['footprint_experiment'] = dict(
            rule='Tangent standard deviations unchanged; normal standard deviation uses reference spacing / 4',
            shared='positions, normals, tangent sigma, live support/opacity, pin latch, camera, lighting, image normal filter',
            scope='Appearance-only paired experiment. Reduced coverage can expose missing support; no physical-quality or rest claim.',
            helper_sha256=hashlib.sha256((support_file.parent/'footprint_policy.py').read_bytes()).hexdigest())
        metadata['footprint_experiment']['diagnostics_sha256'] = hashlib.sha256(
            (support_file.parent/'footprint_diagnostics.py').read_bytes()).hexdigest()
        metadata['render_source_sha256']['settled.py'] = hashlib.sha256(
            (support_file.parent/'settled.py').read_bytes()).hexdigest()
        metadata['raster_backend'] = 'legacy'
        metadata['raster_files'] = raster_identity('legacy')
        metadata['footprint_experiment']['all_observations_valid'] = all(
            row['observation']['valid'] for row in footprints)
        args.out.with_suffix('.footprints.json').write_text(json.dumps(dict(
            upper_cohort=dict(raw=cohort_index, y_min=upper_floor, definition='fixed source IDs in upper target-height band'),
            rows=footprints), indent=2))
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
