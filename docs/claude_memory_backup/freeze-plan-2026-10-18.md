---
name: freeze-plan-2026-10-18
description: User decision 2026-10-15 ~13:40 CDT — freeze on Sat 10/18 as D135 + D146b + D149 (+ floating-Gaussian fix only if it passes its gate by then); WL (D148) not adopted; algorithm-dependent evidence redone on that tag by 11-07
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-15T23:03:00.634Z
---

User, 2026-10-15: "정 안 되면 우리 gradient visualization 했던 거로 fix 해 버리자 pipeline". In the follow-up question the user chose:
- **Fallback freeze:** RGV/deliverable code (D135 + D144 display) **+ D146b + D149**.
- **Deadline:** the floating-Gaussian fix gets until **Sat 10/18**.

**Context:**
- D148 WL was not adopted. The user: "두 영상에서 짜피 모두 다 Gaussian이 보여버려서", meaning both D146b and WL still show floaters.
- The visible floaters are 1–3-particle pieces, dragon only (5–11 per run). u detaches them. Inside holes the cleanup's nearest target point lies across the hole. No control reaches a piece inside the body's MPM cell.
- D149 (symmetric spacing push) passed bunny and dragon at 300k.
- D150 (angular, drag affine = old D99) is timeboxed to 10-16 evening and left out if not clean.

**How to apply:**
- Branch refreeze-cand gets D146b + D149 defaults on; tag on 10/18.
- Then the D145 post-tag queue runs: B1–B3, B7 (rgv_patch2), B8–B10, and visible_apart on every run.
- RGV, the D147 scene and the B13 production runs also wait for the tag.
- No algorithm-dependent runs on the deliverable code until then.
- Exact (bit-for-bit) runtime speed-ups may land after the tag without redoing the evidence.

See [[siggraph-target]], [[d137-cell-from-n]], [[showcase-throw-scene]].

**State 2026-10-15 ~14:10 CDT:**
- refreeze-cand is pushed at 169f251, untagged. Suite: 412 passed.
  - Defaults: relax_driven, u_adam_carry and spacing_symmetric on. The --no- forms give the 09b path.
  - Pinned refs: tests/test_refreeze_reference.py, including a deterministic pipeline digest that holds on hyde06's cards only.
- D150 (--drag_affine, --layer_torque_free) is NOT adopted. Its code is on wip/d149, default off.
  - Reason: making angular momentum exact raises the visible net rotation 7–8×, because a shape change turns the body at zero L.
- D145 post-tag queue is ready: 446–471 runs, ~75 card-hours, including amendment 6 (ablations NRD/NUC/NSS/NFX).
  - Launch: `posttag_local.sh TAG`, then `POSTTAG_FX_FLAG=... posttag_setup.sh TAG`.
- Floater fix (user approved "진행" at ~14:05): `--layer_link`.
  - Definition: the k_layer_project step (u + relaxation) may not push a particle's nearest frozen neighbour (spacing list) beyond max(berth, its distance before the step).
  - Measured basis: the layer step opens the link in 85–97 % of blob detachments.
  - Gate runs tonight; read Friday morning. The user decides adoption Friday/Saturday.
- B13 is rebased onto 169f251 (49d8706). It merges after its CUDA tests, proof and suite pass.

**State 2026-10-15 afternoon (D151 = `--layer_link`):**
- Code: wip/d151 dd3d959 + bc1feac on 169f251, default off; server copy repo_d151.
  - New kernel k_layer_link_take decides the step 0/1 outside the tape. k_layer_project_take and its hand adjoint apply it.
  - Frozen kernels untouched; the off path launches them.
  - Raises an error without min_spacing. test_frozen_recipe now lists layer_link False.
- CPU tests: 21 passed. CUDA tests run in the full suite (tmp/d151_suite.sh, records the exit status). The gate chains start only on suite exit 0.
- Gate pre-registered as D151 at 14:02 CDT (server clock):
  - NL/NP dragon + bunny, seed 97, against SL/SP (SL2/SP2).
  - Base side renders queued for the side-by-sides.
  - Read Friday morning.
- 14:11 CDT: full suite on repo_d151 passed (418 passed, 2 skipped; pinned refreeze digest included, so the off path is bit-for-bit).
  - Pre-registration commit 555c42e; launch notes 7e5b644.
  - Dragon chain NL started on GPU 1 at 14:12. Bunny chain follows once CDX's GPU 3 render frees.
  - ETA ~16:00, then the base SL/SP side renders.
  - Read moved to TONIGHT (reader scripts/probes/d151/d151_gate_read.py), giving the user Friday to decide.
- **18:00 CDT, after the local restart.**
  - The runs finished rc 0, but every post-run step found no frames file. repo_d151 writes the full archive (`*_render_full_dt_iso_nn.npz`, 13–31 GB); D149's `_frames12.npz` writer was a repo_d149s-only patch.
  - Fix: `tmp/d151/frames12_from_archive.py` + `tmp/d151/d151_fix.sh` (agent D151 locks). They rebuild frames12 from the archives with the same rule and rerun the chain's post-run block. The runs are not repeated.
  - Preliminary read (`d151_gate_read.py`):
    - (5) cost: OK.
    - (3) end: OK, except dragon NP thin 3.29 % against 3.25 %.
    - (2) early: fails only against zero-width bands, by about 1e-4 relative.
    - **(4) momentum: FAILS for real.** Dragon NL centre of mass drifts steadily to 0.0025 a (SL 0.0001 a, about 18×); record body_net is 10–35× the base on all four runs. Cause: the link withholds outward steps per particle, which breaks u's rigid-free projection (D94) and the relaxation's symmetry (D149).
  - (1) visible pieces: pending (post.sh picks it up).
- Full archives (89 GB) kept until the gate is read, then delete them (log in cleanup.log).
- B13: suite run 1 had 3 failures in test_runtime_ab (the stub lacked `_zd`). Fixed in 251d901 (wip/b13), deployed, suite rerun via `tmp/b13/b13_suite2.sh` (exit in `output/gpu/b13/suite_251d901.exit`).
  - The deterministic proof was already SAME; the 40k smoke runs F/FM ×4 scales are done.
  - Merge into refreeze-cand after exit 0.
- RGV relabelled dragon 1080p fetched to `output/results_2026-10-14/render_gradient_viz/v2_dragon/` (16:53 server render). The 4K renders continue via rgv_queue3.sh.
- CDX: u opens the link in 72–80 % of the 198 early crevice blob detachments; the clamp bites 69–72 %; nearest neighbour within 1.5 a in 82–85 % with the clamp (20–21 % without).

**FROZEN 2026-10-15 ~18:55 CDT: tag `freeze-2026-10-15` = fc02a1a on refreeze-cand (pushed).** The user chose "지금 고정" after D151 failed.
- Contents:
  - D135 + D146b + D149 as defaults;
  - B13 merged (default off; wip/b13 251d901);
  - the post-tag probe scripts (visible_apart, rgv_patch, rgv_patch2).
  - No floating fix.
- D151 verdict: NOT adopted. Committed on settled-base c19d4fb.
  - (1) dragon unique side blobs −33 % (it needed −50 %).
  - The detector confirmed 8 pieces against 6, including 2 new late ones.
  - (4) the centre of mass creeps 18×.
- Server:
  - repo_t / repo_ts / repo_tgf were built by `posttag_setup.sh freeze-2026-10-15` (all patches matched).
  - The determinism gate `detproof_tag` sits at the head of main.queue. posttag_watch.sh appends 446 items when it passes: B1s97 12, A6 75 (no NFX), B7 2, B1s 24, B1x 8, B2 72, B8 20, B9 20, B10 4, B3 190, B1full 19.
  - tmp/d145/max_cards = 3.
  - D151's full archives (89 GB) were deleted (logged). Output is at 317 GB.
- Agents launched for the post-tag work:
  - D147 final scene: merge the tag into wip/d147, run the suite, then s97 / twin / s99 at 300k plus renders → output/results_2026-10-15/d147_throw_final/.
  - B13 300k production on repo_ts.
- The user's view (19:00): fix the floaters "with additional post-processing if needed"; asked whether it matters for SIGGRAPH. My answer: report them honestly; any display filter must be a disclosed rule applied to every method, with unfiltered videos in the supplementary. See [[physical-detail-not-display-trick]].
