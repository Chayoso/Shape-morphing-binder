# Render influence (agent RI): resume note, 2026-10-22 21:10 CDT

Server scripts: `tmp/render_influence/` (copies of the files in this folder). Results: `output/gpu/render_influence/` and
`output/gpu/rgv_held/`. Local: `output/results_2026-10-22/render_influence_tag/` (README_ko.md, `rgv_tag_vs_held/`).

## Done
- Panels 1, 2 and 4 in the RGV studio look (`ri_figs.py`), README_ko.md, one line in output/README.md.
- End-surface numbers (`ri_endmap.py`, data/endmap_*.json). The white-page fig3 was removed; the user did not like it.
- RGV 2×2 "tag | render weight held" (gpu1 queue job 01, `01_ri_rgv_held_bunny.sh`):
  - bunny300k_RH97 = repo_tgf + `--lambda_ema 0`;
  - stills and videos copied locally to `rgv_tag_vs_held/`;
  - the dumps and frame folders were deleted (cleanup.log). `bunny300k_RH97_viz.npz` (1.6 GB) and `_frames12.npz` (0.6 GB) are kept for re-renders.
- Panel-3 pass 1 (CPU): `output/gpu/render_influence/video/vals_{bunny,dragon}_{d,e2}.npy` + `.json`.

## Waiting
- **gpu1 queue job `80_ri_panel3_video.sh`**: after 03_d162_d162d and 21_d164_pilot_gates.
  - It runs `ri_video.py` (RGV look, imported renderer) for bunny then dragon: 1080p front/side, plus 2×2 full/half.
  - Then the mp4s in `output/gpu/render_influence/video/`.
  - Status lines go to `output/gpu/render_influence/status.txt` ("RI render MESH ... rc", "RI pass 2 done").
  - **ri_video.py has never run (no card was free to test it).**
- `ri_video_after.sh` (PID in `tmp/render_influence/after_video.pid`, CPU) waits for "RI pass 2 done". It then runs:
  - `ri_sheet.py`: `output/gpu/render_influence/fig3_hero_sheet.png`;
  - `ri_figs.py`: the combined figure, with the sheet as panel 3.

## On resume
1. `tail tmp/gpu1_queue/runner.log`; `tail output/gpu/render_influence/status.txt`.
2. If job 80 failed:
   - read `output/gpu/render_influence/video/render_bunny.log` and fix `tmp/render_influence/ri_video.py`;
   - smoke-test with `+frames=20-22`;
   - re-drop the job as `81_ri_panel3_video.sh` (write it as `.tmp` first, then `mv`).
3. Copy to local `output/results_2026-10-22/render_influence_tag/`:
   - `video/*.mp4` into `video/`;
   - `fig3_hero_sheet.png`, `fig_render_influence_all*.png`.
   Then fill the README "영상" section with the per-mesh dmax and the colour story.
4. Cleanup after copying (log each in output/gpu/cleanup.log):
   - `output/gpu/render_influence/video/frames/`;
   - `vals_*.npy` (~1 GB);
   - `endmap_*_vals.npz`.
