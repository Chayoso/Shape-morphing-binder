# D173 resume (agent D173, started 2026-10-29 09:40 CDT; this file 11:55 CDT)

Read ONE file: output/gpu/D173_SUMMARY.md (tmp/d173/d173_summary.py rewrites it; every reader calls it at its end).

## Code (repo_d173 = wip/d173 c1855865, pushed; files copied to the server, repo_d173 is not a git checkout)
- 98303b79 texture appearance; 0ac73e72 --colour_kernel nearest; 3c447081 split-run telemetry;
- c1855865: each block's render share recorded on EVERY run (window/solve.py _calibrate_lambda: g_share_u, g_share_dfc after
  PCGrad, lam_block_u / lam_block_dfc as applied; run/runner.py records); --appearance stripes (appearance.py).
  Gate: queue job 209_d173_gate_t.sh -> output/gpu/d173/gates/V3t.ok / V3t.fail (defaults vs the tag, band vs repo_d172,
  Hu vs repo_d172 X2b, the stripes smoke, the records present). EVERY later D173 job waits for V3t (job_common.sh gate_t).
  The pre-c1855865 solve.py is kept at tmp/d173/stage/solve_3c447081.py.

## Sections of D173_SUMMARY.md and what feeds them
1. Real scales representability (done: nothing <= ~600k passes; 1.94M / 3M pass; densification HELD for the user).
2. Colour edge (done) + --colour_kernel nearest: band 40k PASS (edge 1.02 / 0.99, r 0.91 / 0.92); band 300k s97 FAILS by the
   letter (edge 2.93 vs Rx 2.68, r 0.83 vs 0.92) but stopped at window 42 (7.0 s) where Rx ran 82 (13.7 s): equal-time edges
   read by output/gpu/d173/auto/band300time (run.sh PID run.pid, d173_edge2.py raw=1681 / 2441 / 3281).
3. Bob (texture): Rx +2.3-2.5 dB over the twin, 9-12 dB below the ceiling. Rxn 40k (auto/tex40n): kernel PSNR = Rx's,
   nearest-colour PSNR +0.35 / +0.54 dB over Rx; s99 thin +0.053 over the twin (shape gate 0.05).
   Pending: job 218 (tex300k_Rxn_97) -> after_tex300n.sh; job 219 (40k Rx / Rxn reruns with the block records) ->
   after_tex40t.sh (auto/tex40t, rerun_cmp.txt); job 222 (300k Rx rerun) -> after_tex300t.sh (auto/tex300t).
4. Attribution, stylized scales E, 300k: job 213 (Hs_97, Hm_97; twin d172 scl300k_T_97) -> after_attr.sh 97; job 214
   (T_99, Hs_99, Hm_99) -> after_attr.sh 99; job 215 u-off arm (atr300k_Hu0_97: Hu --u_off) -> after_uoff.sh (re-reads attr97
   with 4 arms; deliver/attr97_uoff sheet). Criterion (pre-set): share end >= 0.40 in Hs, < 0.10 in twin and Hm.
5. Option C (the dragon's own scales magnified K x, d173_magscale.py): probes output/gpu/d173/assets/magC/probe_table{2,3,4}.txt.
   K 2 / 2.5 / 3 (the user's range) read corr 0.088 / 0.118 / 0.143 at 300k, 0.109 / 0.149 / 0.161 at 550k; K 4 / 5 0.164 /
   0.221; K 6 / 7 / 8 0.265 / 0.326 / 0.377 -> variant.txt "6 6.0 24" (K 6, s_bg 24 pitches). The first K 2.5 mesh had a dot
   in its name (orientation.asset_name cut it: read unrotated) -> renamed M25, records in assets/magC/turned25/.
   Jobs 216 / 217 (mag300k_Hk_97 / 99: Hu with the K 6 pictures) -> after_magC2.sh 97 / 99 (auto/magC97, magC99: detail read
   s_bg 24, d173_attr.py, the close-up sheet, twin | full video 4K + 1080p in deliver/magC97, magC99).
6. Bob with nearest colour: see 3.
7. Stripes (the coordinator's new colour scene): pre-run check tmp/d173/d173_layout_check.py -> output/gpu/d173/layout/check.json
   (twin layout vs stripes kappa 0.00-0.01, 0.02-0.03 on the outer layer; band 0.68 / 0.78; checker -0.02 / 0.05-0.10).
   Jobs 220 (40k T / Rxn x 97 / 99) -> after_strp.sh 40; 221 (300k T / Rxn s97) -> after_strp.sh 300 (colour r, edges
   d173_edge2.py, PSNR / kappa / shares d173_stripread.py, deliverables deliver/strp40, strp300 via deliver_colour2.sh).

## Readers waiting (PIDs in tmp/d173/*.pid; stop only by these PIDs)
after_attr97 / 99, after_uoff, after_magC97 / 99 (after_magC2.sh), after_tex300n, after_tex40t, after_tex300t,
after_strp40, after_strp300.

## Queue order (tmp/gpu1_queue, shared with D174 / D175 / D176; never gpu2)
209_d173_gate_t, 213 attr97, 214 attr99, 215 u-off, 216 / 217 option C, 218 Bob 300k Rxn, 219 Bob 40k reruns, 220 / 221
stripes, 222 Bob 300k Rx rerun.

## Disk
447 GB at the start (tmp/d173/disk_base.txt); runs stop at 495 (job_common.sh disk_ok).

## Resume
- A failed run: delete its .rc and rerun its job script. The summary: `$PY tmp/d173/d173_summary.py output/gpu/D173_SUMMARY.md`.
- If V3t fails: read output/gpu/d173/auto/209_gate_t/summary.md and gates/cmpt*.txt / keys_t.txt; fix; requeue the gate
  and the jobs (they exited on V3t.fail).
- Never pkill / pgrep -f.
