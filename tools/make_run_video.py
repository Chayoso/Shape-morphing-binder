#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


FRAME_KINDS = {
    "render": ("ep*/render.png", "render.mp4"),
    "alpha_primary": ("ep*/alpha_primary.png", "alpha_primary.mp4"),
    "alpha_worst": ("ep*/alpha_worst.png", "alpha_worst.mp4"),
    "viz": ("viz/ep*.png", "viz.mp4"),
    "views": ("viz/ep*_views.png", "views.mp4"),
}


def collect_frames(run_dir: Path, kind: str) -> list[Path]:
    pattern, _ = FRAME_KINDS[kind]
    return sorted(run_dir.glob(pattern))


def build_video(run_dir: Path, kind: str, fps: int, ffmpeg_bin: str, overwrite: bool) -> Path | None:
    frames = collect_frames(run_dir, kind)
    if not frames:
        print(f"[skip] {run_dir}: no {kind} frames")
        return None

    _, default_name = FRAME_KINDS[kind]
    out_path = run_dir / default_name
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists")
        return out_path

    with tempfile.TemporaryDirectory(prefix="codex_video_") as tmp:
        tmp_dir = Path(tmp)
        for i, src in enumerate(frames):
            dst = tmp_dir / f"frame_{i:05d}.png"
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[ok] {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble mp4 videos from run PNG frames.")
    ap.add_argument("runs", nargs="+", help="Run directories or globs")
    ap.add_argument("--kind", choices=sorted(FRAME_KINDS.keys()), default="render")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for raw in args.runs:
        matches = sorted(Path().glob(raw)) if any(ch in raw for ch in "*?[]") else [Path(raw)]
        for p in matches:
            if p.is_dir():
                run_dirs.append(p)

    if not run_dirs:
        raise SystemExit("No run directories found.")

    for run_dir in run_dirs:
        build_video(run_dir, args.kind, args.fps, args.ffmpeg, args.overwrite)


if __name__ == "__main__":
    main()
