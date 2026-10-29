# D173 resume (agent D173, started 2026-10-29 09:40 CDT)

Read ONE file: output/gpu/D173_SUMMARY.md (tmp/d173/d173_summary.py rewrites it; run it by hand after any new read).

## The three tasks
1. The dragon's REAL scales: representability first (no runs). tmp/d173/d173_probe.py on the pipeline's own operators: a
   perfect sample of the detailed dragon (layer), its exterior field (zb, radius r, the lattice of target.py), the loss's
   shading pictures against the true surfaces' (D159 mesh shells): corr per lever (N, surface density F, radius r, px).
   Gate: picture corr >= 0.25. probe_a (300k: r 3 / 2 / 1.5 x 188 / 256 / 384 px, F 2 / 4) and probe_b (500k, 1M, 1.5M)
   -> output/gpu/d173/probe/*.jsonl. Runs only if a setting within 500k passes (none of the 300k ones does).
   The r = 1.5 OOM: the field's zero set breaks into interior pockets (43 % of the discs deeper than 3 pitches, 1.9M discs
   at 188 px vs 175k at r = 3) -- the field, not a memory bug; its field corr is 0.
   Surface density: no sampler option in the current code (D122's --surface_density lives only on old branches, refuted
   2026-10-07); the probe emulates it (the outer band F times denser).
2. Colour edge: tmp/d173/d173_edge.py -> output/gpu/d173/edge/edge.jsonl. The kernel (render/exterior.py:400-413
   found_disc_albedo, R = 3 pitches) widens the target's sharp paint to 2.1-2.3 pitches; the arm's own particle layout is
   2.2-2.8 pitches wide. Display fix (both target and arms): tmp/d173/d173_studio.py +colour=nn (each disc its nearest
   particle's colour). The loss is unchanged (a narrower colour kernel in the loss = the next definition, not made).
   Deliverables: output/gpu/d173/deliver/band300 (kernel / nn painted ends, grey ends, nn video), cmb300 (job 201 + reads).
3. Texture: Keenan Crane's Bob (output/gpu/d173/assets/d173bob_tex.obj = assets/bob.obj's vertices + bob_tri's UVs,
   d173bob_diffuse.png). Code: repo_d173 = wip/d173 98303b79 (appearance.TextureMap, --appearance texture --texture
   OBJ,PNG; prepare returns the frame). Ceiling: tmp/d173/d173_texprobe.py -> output/gpu/d173/texprobe/texprobe.jsonl
   (40k 26.6 dB / 0.972, 300k 28.0 dB / 0.970 held-out; flat colour 16.2 / 15.0 dB). Runs: job 200 (gate + 40k twin / Rx x
   97 / 99), job 202 (300k twin / Rx); reads tmp/d173/d173_texread.py -> output/gpu/d173/auto/tex40, tex300.

## Queue (tmp/gpu1_queue; our runner on card 3, PID tmp/d173/runner_card3.pid; cards 0 / 1 held by agent D174;
## tmp/d173/card_waiter.sh starts our runners there when they free; never gpu2)
- 195 probe_a, 197 probe_b, 200 gate + texture 40k, 201 combined relief + paint 300k, 202 texture 300k.
- Detached on card 3 (beside the queue's job): tmp/d173/after_gpu.sh tex40 / cmb300 / tex300 (reads + deliverables,
  PIDs tmp/d173/after_*.pid), tmp/d173/deliver_colour.sh band300 (done for the ends; its nn video).

## Disk
- 447 GB at the start (tmp/d173/disk_base.txt); runs stop at 495.

## Resume
- A failed run: delete its .rc and rerun its job script. The summary: `$PY tmp/d173/d173_summary.py output/gpu/D173_SUMMARY.md`.
- Stop: kill the runner / waiter / after PIDs (their own processes only). Never pkill / pgrep -f.
