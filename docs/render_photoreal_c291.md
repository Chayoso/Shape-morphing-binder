# Mixed60 studio Gaussian render

This is a presentation render of the existing mixed60 trajectory, not another
simulation. Source: `c291_bunny_mixed60_render_full_dt_iso_nn.npz`, produced at
bunny N=300k, T=20, dt=1/240, dx=0.3062907544 wu, loss grid 36^3, seed 1,
animations=300 and diagnostic cap=60. The simulation and its thin-feature
limitations are described in `body_control_p291.md`.

`scripts/render_splat_photoreal.py` renders a native 1920x1080 single view at
azimuth 35/elevation 18 degrees. It uses the same raw delivered frames and stride 12
as the existing two-view 450px export, at 20 fps. Every saved position is passed
unchanged to the Gaussian rasterizer. The studio appearance is a synthetic
uniform satin ceramic material with fixed GGX roughness 0.36, three fixed lights
and a smooth backdrop. There is no captured surface texture or recovered real
illumination. No mesh reconstruction, added geometry, hole filling or frame-wise
exposure fitting is used.

## Geometry and compute contract

The existing renderer's density normals, two 32-neighbour smoothing passes,
8-neighbour support, opacity 0.92 and adaptive capped disc radii are retained.
Compositing uses the original unblurred coverage; the normal-buffer smoothing
does not expand it. `SettledAppearance` freezes active-pin normal/radius values,
while density support stays live. Every raw delivered frame is checked for
movement after admission, including frames omitted by stride 12; any violation
refuses export rather than hiding the movement.

CUDA performs kNN, the density grid, normals, covariance decomposition,
rectangular rasterization and shading. CPU responsibilities are archive/image
I/O, small configuration metadata and orchestration. PNG buffers cross to the
host at the export boundary. Video encoding uses NVIDIA H.264 NVENC. A snapshot
of dependencies is deployed into an isolated `/data/.../work/gpu_refactor/render1080`
directory so concurrent pipeline refactoring cannot change an active render.

## Reproduction

Run only on hyde06 via hyde01, with CUDA_VISIBLE_DEVICES=2 and caches/TMPDIR under
`/data`, after the usual GPU occupancy check and 45-second launch spacing:

```sh
python scripts/render_splat_photoreal.py \
  /data/relcfd/chayo/physmorph_v2/output/c291/c291_bunny_mixed60_render_full_dt_iso_nn.npz \
  /data/relcfd/chayo/physmorph_v2/work/gpu_refactor/render1080/mixed60_studio_1080p.mp4 \
  --frames-dir /data/relcfd/chayo/physmorph_v2/work/gpu_refactor/render1080/frames
```

The output JSON records raw frame indices, dimensions, camera, material,
timings, source configuration and script SHA256. `snapshot_sha256.json` records
the exact deployed dependencies. Local deliverables are under
`output/c291/photoreal1080/`.

## Validation

Two CPU tests verify that transient pin movement between rendered frames refuses
export, while movement before admission and movement of unpinned particles remain
valid. The renderer's code was independently reviewed for geometry alteration,
coverage inflation, pin handling, perspective consistency and GGX view vectors.
The final export completed on hyde06 GPU 2 (RTX 6000 Ada) on 2026-09-26 in
**35.14 seconds**, including setup and NVENC encoding. `ffprobe` verifies
1920x1080, H.264, 20 fps, 67 frames and 3.35 seconds. All 67 encoded frames were decoded
and visually inspected in six contact sheets, with full-resolution inspection of
raw 84, 144, 228 and 781. No camera clipping, black/corrupt frames or added crossfade
were found. There is no texture to evaluate for material-coordinate sliding.
The early head/ear growth front remains soft and partly translucent; ear tips
retain sparse fringes during growth, and the delivered shape is rounded and
under-detailed. These limitations are visible in the footage, not repaired by
shading. This presentation cannot certify absence of physical holes or rest of
all unpinned particles.

The first 300k GPU covariance decomposition exposed a cuSOLVER batch-size error.
The shared helper now processes bounded 8192-row float64 eigensolve batches;
the successful final export includes that repair and the CUDA kNN self-first
correction. Both changes retain the covariance and support definitions. Final
dependency hashes are in `output/c291/photoreal1080/snapshot_sha256.json`.

## 4K export

At the user's follow-up request, the same stable render snapshot was run at
native **3840x2160** on hyde06 GPU 2. Camera, material, raw frames, stride 12,
support and pin policies were unchanged. This run was isolated from the
concurrent pipeline/CuPy refactor; the script, kNN and covariance dependency
hashes were checked against the completed 1080p snapshot before launching.

The result is `output/c291/photoreal4k/mixed60_studio_4k.mp4`: H.264 NVENC,
20 fps, 67 frames, 3.35 seconds, 672696 bytes. Total measured execution was
**46.35 seconds**, including 1.57 seconds setup and 1.43 seconds encoding.
All 782 delivered raw frames passed pin-motion validation. Project storage was
96.119 GB after rendering, below the user's 100 GB cleanup threshold; the
existing storage guard remained active.

All 67 encoded 4K frames were decoded and inspected across six contact sheets;
the ear regions at raw frames 228 and 781 were also inspected in native-pixel
crops. There was no observed clipping, corrupt frame or added crossfade.
The same sparse/translucent growth fringes and rounded, under-detailed final
shape remain. Increasing output resolution does not repair those geometric
limitations or establish all-particle rest. The video SHA256 is
`6e164e4ee2e697d26000dd7ecd48bdd5cc7e03168a345195a9690465ff3a6428`;
local metadata, `snapshot_sha256.json` and `visual_qa.json` accompany the video.
