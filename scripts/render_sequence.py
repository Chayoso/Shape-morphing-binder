"""Render every saved commit with the viewer's anisotropic 3DGS model.

This is a deliverable-QA helper, not a metric path.  Color is fixed in source/material
coordinates so a contact sheet also reveals texture swimming or state/frame mismatch.
Requires a CUDA 3DGS rasterizer (run on hyde06).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physmorph.render.covariance import sigma0_from_nn
from physmorph.render.children import (expand_children_numpy,
                                       tangent_child_offsets)
from physmorph.render.photoreal import render_3dgs
from physmorph.render.support import MaterialSupport


def material_colors(source: np.ndarray) -> np.ndarray:
    lo = source.min(0)
    span = np.maximum(source.max(0) - lo, 1e-6)
    uvw = np.clip((source - lo) / span, 0.0, 1.0)
    # Warm/cool material-space bands remain attached to particle identities.
    bands = 0.5 + 0.5 * np.sin(7.0 * uvw[:, :1] + 4.0 * uvw[:, 1:2])
    warm = np.array([0.86, 0.43, 0.20], np.float32)
    cool = np.array([0.16, 0.48, 0.78], np.float32)
    return np.ascontiguousarray(bands * warm + (1.0 - bands) * cool, np.float32)


def load_render_archive(path: str | Path, surface_frac: float = 0.50,
                        sigma_scale: float = 1.0, child_count: int = 1,
                        child_sigma_scale: float = 0.55,
                        child_offset_scale: float = 0.35) -> dict:
    """Load and validate a saved run without importing or invoking the CUDA renderer."""
    required = {"frames", "F_samples", "F_sample_idx", "src", "tgt"}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"render archive is missing: {', '.join(missing)}")
        out = {key: np.asarray(data[key]).copy() for key in required}
        out["render_mask"] = (np.asarray(data["render_mask"]).copy()
                              if "render_mask" in data else None)
        out["target_render_mask"] = (np.asarray(data["target_render_mask"]).copy()
                                     if "target_render_mask" in data else None)
        out["sigma0"] = float(np.asarray(data["sigma0"])) if "sigma0" in data else None
        out["source_child_offsets"] = (np.asarray(data["source_child_offsets"]).copy()
                                       if "source_child_offsets" in data else None)
        out["target_child_offsets"] = (np.asarray(data["target_child_offsets"]).copy()
                                       if "target_child_offsets" in data else None)
        out["gauss_child_sigma_scale"] = (
            float(np.asarray(data["gauss_child_sigma_scale"]))
            if "gauss_child_sigma_scale" in data else float(child_sigma_scale))

    source, target = out["src"], out["tgt"]
    frames, Fs = out["frames"], out["F_samples"]
    indices = out["F_sample_idx"].astype(np.int64, copy=False)
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 2:
        raise ValueError(f"src must be a non-empty (N,3) cloud, got {source.shape}")
    if target.ndim != 2 or target.shape[1:] != (3,) or len(target) < 2:
        raise ValueError(f"tgt must be a non-empty (M,3) cloud, got {target.shape}")
    if frames.ndim != 3 or frames.shape[1:] != source.shape:
        raise ValueError(f"frames must have shape (K,{len(source)},3), got {frames.shape}")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("F_sample_idx must be a non-empty vector")
    if Fs.shape != (len(indices), len(source), 3, 3):
        raise ValueError("F_samples must have shape (len(F_sample_idx),N,3,3)")
    if np.any(indices < 0) or np.any(indices >= len(frames)):
        raise ValueError("F_sample_idx is outside the saved frame sequence")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("F_sample_idx must be strictly increasing")
    if not (np.isfinite(source).all() and np.isfinite(target).all()
            and np.isfinite(frames[indices]).all() and np.isfinite(Fs).all()):
        raise ValueError("render archive contains non-finite selected state")

    from physmorph.pipeline.runner import _surface_weights
    source_mask = (out["render_mask"].astype(bool) if out["render_mask"] is not None
                   else (_surface_weights(source, 24, surface_frac, 0.05) > 0.5))
    target_mask = (out["target_render_mask"].astype(bool)
                   if out["target_render_mask"] is not None
                   else (_surface_weights(target, 24, surface_frac, 0.05) > 0.5))
    if source_mask.shape != (len(source),) or target_mask.shape != (len(target),):
        raise ValueError("source/target render masks do not match their clouds")
    if source_mask.sum() < 1 or target_mask.sum() < 2:
        raise ValueError("render masks contain too few source/target samples")
    sigma0 = (out["sigma0"] if out["sigma0"] is not None
              else sigma0_from_nn(target[target_mask], sigma_scale))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be finite and positive")
    source_offsets, target_offsets = (out["source_child_offsets"],
                                      out["target_child_offsets"])
    if (source_offsets is None) != (target_offsets is None):
        raise ValueError("source and target child offsets must be archived together")
    if source_offsets is None:
        source_offsets = tangent_child_offsets(source, source_mask, sigma0, child_count,
                                               child_offset_scale)
        target_offsets = tangent_child_offsets(target, target_mask, sigma0, child_count,
                                               child_offset_scale)
    if (source_offsets.ndim != 3 or source_offsets.shape[0] != len(source)
            or source_offsets.shape[2] != 3):
        raise ValueError("source_child_offsets must have shape (N,C,3)")
    children = source_offsets.shape[1]
    if children < 1 or children > 4 or target_offsets.shape != (len(target), children, 3):
        raise ValueError("target child offsets must match source child count in [1,4]")
    if not np.isfinite(source_offsets).all() or not np.isfinite(target_offsets).all():
        raise ValueError("child offsets must be finite")
    child_sigma_scale = (1.0 if children == 1
                         else out["gauss_child_sigma_scale"])
    if not np.isfinite(child_sigma_scale) or not 0 < child_sigma_scale <= 1:
        raise ValueError("gauss_child_sigma_scale must be in (0,1]")
    out.update(F_sample_idx=indices, render_mask=source_mask,
               target_render_mask=target_mask, sigma0=float(sigma0),
               source_child_offsets=np.ascontiguousarray(source_offsets, np.float32),
               target_child_offsets=np.ascontiguousarray(target_offsets, np.float32),
               gauss_child_sigma_scale=float(child_sigma_scale))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True, help="animated GIF output")
    ap.add_argument("--frames_dir", default="", help="PNG directory (default: <out>_frames)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--views", default="0.6,2.2")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--surface_frac", type=float, default=0.50)
    ap.add_argument("--gauss_children", type=int, default=4)
    ap.add_argument("--gauss_child_sigma_scale", type=float, default=0.55)
    ap.add_argument("--override_child_sigma_scale", type=float, default=0.0,
                    help="render-only footprint ablation; 0 reuses the archived forward model")
    ap.add_argument("--gauss_child_offset_scale", type=float, default=0.35)
    ap.add_argument("--no_support_opacity", action="store_true",
                    help="disable target-free material-support opacity (A/B only)")
    args = ap.parse_args()

    data = load_render_archive(args.npz, args.surface_frac, args.sigma_scale,
                               args.gauss_children, args.gauss_child_sigma_scale,
                               args.gauss_child_offset_scale)
    if args.override_child_sigma_scale > 0:
        if args.override_child_sigma_scale > 1:
            raise ValueError("override_child_sigma_scale must be in (0,1]")
        data["gauss_child_sigma_scale"] = float(args.override_child_sigma_scale)
    frames = data["frames"]
    F_samples = data["F_samples"]
    indices = data["F_sample_idx"]
    source = data["src"]
    target = data["tgt"]
    source_mask = data["render_mask"]
    children = data["source_child_offsets"].shape[1]
    colors = np.repeat(material_colors(source)[source_mask], children, axis=0)
    support = None if args.no_support_opacity else MaterialSupport.from_rest(source, 8)
    sigma0 = data["sigma0"]
    child_sigma0 = sigma0 * data["gauss_child_sigma_scale"]
    azimuths = [float(v) for v in args.views.split(",")]
    out = Path(args.out)
    frame_dir = Path(args.frames_dir) if args.frames_dir else out.with_suffix("").with_name(
        out.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    center = target.mean(0)
    radius = float(np.linalg.norm(target - center, axis=1).max())
    dist = max(radius * 3.0, 1e-3)

    faded_max = 0
    for j, (idx, F) in enumerate(zip(indices, F_samples)):
        opacity = 0.92
        if support is not None:
            support_alpha = support.opacity(frames[idx])
            faded_max = max(faded_max, int((support_alpha[source_mask] < 0.5).sum()))
            opacity = np.repeat(0.92 * support_alpha[source_mask], children)
        child_x, child_F = expand_children_numpy(
            frames[idx], F, data["source_child_offsets"], source_mask)
        panels = [render_3dgs(child_x, colors, F=child_F, sigma0=child_sigma0,
                              opacity=opacity,
                              azimuth=az, elevation=0.28, dist=dist, res=args.res,
                              center=center, device="cuda")
                  for az in azimuths]
        rgb = (np.clip(np.concatenate(panels, axis=1), 0.0, 1.0) * 255).astype(np.uint8)
        im = Image.fromarray(rgb)
        ImageDraw.Draw(im).text((7, 6), f"commit {j:03d}  frame {idx:04d}",
                                fill=(25, 25, 25))
        im.save(frame_dir / f"frame_{j:04d}.png")
        rendered.append(im.convert("P", palette=Image.ADAPTIVE, colors=192))

    out.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(out, save_all=True, append_images=rendered[1:], loop=0,
                     duration=max(1, int(1000 / args.fps)), optimize=True)
    support_note = ("support opacity off" if support is None
                    else f"max support-faded primitives/frame={faded_max}")
    print(f"saved {out} and {len(rendered)} PNG frames in {frame_dir}; {support_note}")


if __name__ == "__main__":
    main()
