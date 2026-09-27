# P292: full PIC-off comparison

**Do not promote PIC-off alone.** It improves tip occupancy, local density and
within-window motion, but fails the existing 0.971 silhouette-fit gate and does
not stop the common free material. A rendered comparison is diagnostic evidence,
not a claim that the morph is fixed.

## Matched setup and endpoints

Both arms use the identical archived bunny source/target, N=300000, T=20,
dt=1/240, dx=0.3062907544 wu, loss grid 36 cubed, eight inner iterations,
animations=300 and cap=60. The numerical source hash is identical:
`eee25aec5c769c9037852f1cc49c0a50da0b2b9b816e39c10550e7c12c823dbe`.
Only `commit_pic` changes from true to false. Body control, its terminal brake,
stress, surface-u, layer relaxation, subgrid shifting, rendering guidance and pin
policy remain identical; body RPROP is off. The intervention includes its effects
on subsequent optimization and stopping, not just a single isolated correction.

Baseline accepts 32 of 35 attempts in 567.18 s; PIC-off accepts 40 of 41 in
646.80 s. Its final attempted window is rejected. Endpoints are each arm's final
accepted state. All recorded safety guards are zero.

| Endpoint measurement | Baseline | PIC off |
| --- | ---: | ---: |
| Binary silhouette IoU | 0.973039 | 0.966396 |
| Raw Chamfer, wu | 0.055922 | 0.056193 |
| Global target coverage | 0.998053 | 0.996393 |
| Top target coverage | 0.989877 | 0.966366 |
| Top mean density | 0.90748 | 0.97712 |
| Top fraction below half density | 0.09174 | 0.06659 |
| Tip-ball count (target: 89) | 53 | 87 |
| Minimum accepted trajectory det F | 0.76790 | 0.80457 |
| Terminal mean speed, wu/s | 0.01409 | 0.00291 |
| Pinned fraction | 0.87972 | 0.96364 |
| Admitted pins checked for later drift | 263917 | 289092 |
| Maximum pin drift, wu | exactly 0 | exactly 0 |

The silhouette decrease is 0.006642 and persists at the common commit 32
(0.973039 versus 0.966269). At that same commit, top coverage is
0.989877/0.964929, density 0.90748/0.96876 and tip count 53/80. This is not merely
an endpoint-time mismatch. A denser top does not mean it covers the full target.
Coverage/density are independent raw point-cloud measurements, not watertightness
tests or rendered images; their definitions are in the
[comparison method](quality_comparison_p292.md).

The baseline tip peaks at 72 at raw frame 377 and retains 53 of those IDs at its
endpoint, losing 19. PIC-off peaks at 88 at frame 777 and retains 87, losing one.
Neither arm has new tip IDs after its first peak. The candidate fills and retains
the tip much better; its late peak also leaves less time in which to lose material,
so the retention ratio alone is not a causal preservation rate.

At equal accepted commits 3/6/10/20/30, PIC-off top density is
0.33513/0.63345/0.78247/0.95792/0.96887, compared with
0.24088/0.58177/0.67977/0.83487/0.90494. Yet its corresponding top coverage is
0.28664/0.59613/0.84293/0.95500/0.96428, compared with
0.33111/0.60684/0.87004/0.98433/0.98922. Local supply/density and spatial completion
must be judged together.

Both arms first cross 0.75, 0.5, 0.25, 0.225 and 0.22 times initial Chamfer at
commits 1, 2, 6, 9 and 10. The 0.215 crossing occurs at baseline 16 versus PIC-off
20, with actual Chamfer gap +2.47e-5 wu. Neither reaches 0.2 times initial Chamfer.
This is one matched pair with no repeated-run uncertainty interval; the earlier
eight-window PIC-off prefix is not a bitwise repeat of this run.

## What improved, and what still moves

The matched late cohort is **1037 material IDs**, sparse at either arm's common
endpoint and unpinned in both at commit 32. It is outcome-selected, distinct from
the earlier 1374-ID body-RPROP cohort and 9314-ID PIC-off-prefix cohort. All 1037
remain unpinned over accepted endpoints 22–32. The 201 raw states contain no null
holds. Motion units use source-native spacing 0.0349885366 wu.

| Same free material, commits 22–32 | Baseline | PIC off |
| --- | ---: | ---: |
| Raw-step displacement median, sp | 0.01947 | 0.01105 |
| Raw-step absolute-normal median, sp | 0.01071 | 0.00373 |
| Raw-step tangent median, sp | 0.01214 | 0.00917 |
| Raw direction reversals | 13894 / 206363 (6.733%) | 10047 / 206363 (4.869%) |
| Per-commit displacement median / p95, sp | 0.18089 / 0.44090 | 0.19601 / 0.55979 |
| Commit direction reversals | 635 / 9333 (6.804%) | 1240 / 9333 (13.286%) |
| Commit-only net/path median | 0.87514 | 0.82010 |

Lower raw-step movement does not mean rest: committed displacement and committed
direction changes remain and are larger in this matched cohort. Normal estimates
are arm-specific frozen endpoint directions, as in the prior audits; total
displacement and direction tests do not depend on that basis. Reversal is a sampled
negative displacement dot product, not proof of a periodic vibration mode.

The pre-treatment source-upper-surface cohort contains 6712 IDs. Its per-commit
median is zero in both arms; p95 falls 0.21743 to 0.19131 sp. However, pin fractions
at interval start/end rise from 0.73808/0.84952 to 0.78382/0.91031. That cohort's
lower motion therefore includes additional pinning and cannot by itself demonstrate
quieter unpinned material.

The candidate's own final-tail audit also finds nonzero movement: 773 endpoint
surface/unpinned IDs, measured over the final approximately 10% of its simulated
raw states, have normal/tangent medians 0.00445/0.01224 sp per archived step and
5.975% sampled direction reversals. This is an endpoint-selected descriptive
population, not a matched comparison, but it does not certify final free-surface
rest. No held padding is used to make that tail appear stationary.

## Phase localization after PIC removal

Accepted windows 23–32 are decomposed into all 20 raw phases using the same 1037
IDs. Phase 20 includes the last rollout step and remaining commit corrections.

| Matched phase measurement | Baseline | PIC off |
| --- | ---: | ---: |
| Phases 1–19 displacement median, sp | 0.01851 | 0.01062 |
| Phase-20 displacement median / p95, sp | 0.17335 / 0.80408 | 0.07930 / 0.30435 |
| Phase-20 absolute-normal median, sp | 0.12697 | 0.01990 |
| Phase-20 tangent median, sp | 0.06713 | 0.06678 |
| Phase-20 share of total raw path length | 40.37% | 31.38% |
| Reversals 19 -> 20 | 6887 / 10370 (66.41%) | 5165 / 10370 (49.81%) |
| Reversals 20 -> next 1 | 6706 / 9333 (71.85%) | 4719 / 9333 (50.56%) |
| Other adjacent-phase reversals | 301 / 186660 (0.1613%) | 163 / 186660 (0.0873%) |
| Raw net/path median | 0.28473 | 0.57575 |

PIC removal substantially reduces boundary normal return. Boundary tangent
movement is almost unchanged and many boundary reversals remain. These observations
fit the [component accounting](motion_accounting_p292.md), but neither experiment
isolates the remaining final physical step from layer and subgrid changes. No
percentage here is an attribution of image flicker to an individual operator.

## The render target never switches to the final target

The complete PIC-off log contains **no** `converged in the render's metric` switch
message. Its final paced-versus-fixed discrepancy is approximately 0.007866,
larger than morph-versus-paced 0.001210; the configured condition is the opposite
inequality. The renderer reference therefore remains the paced cloud throughout.
The final inner rendering loss is not a certificate of final-target fit.

`render_paced` and `render_paced_conv` are enabled, while `plan_sticky` and
`arrive_cap` are off. Arrived particles independently snap their plan images to
nearest target IDs, allowing duplicate assignments. All particles being classified
as arrived does not imply that this cloud equals the full target point distribution.
The optimizer comment claiming eventual `x_int = target` is not a guarantee here.

The first exact accepted `arrived_end_frac == 1.0` is commit 28, not merely a log
rounded to 100%. It remains exactly one through commit 40. A reference latch after
that accepted commit would have twelve further accepted windows in the observed
continuation, plus the rejected attempt 41. At the proposed handoff, however,
87.863% of particles are already pinned; only 36411 remain free. Silhouette IoU is
already 0.966002. A new fixed reference may help that free material, but permanent
pins may limit recovery. Such a latch requires its own controlled experiment.

Rendering supervision remains at 64 by 64 pixels: the 96 by 96 rebuild is scheduled at
half of `animations=300`, beyond this cap-60 run, and no rebuild appears in the log.
`use_gauss_loss=False`. This guidance optimizes physical controls through coarse
silhouette/shading terms; it is not an optimization of the exported 4K Gaussian
attributes. It cannot be assumed to remove high-resolution splat boundaries.

## Evidence and gates

Local evidence is `output/p292/no_pic60.json`, `no_pic60.log`,
`quality_no_pic60.json`, and `phase_no_pic60.json`; matching originals are under
`/data/relcfd/chayo/physmorph_v2/work/p292`. The isolated audit snapshot is
`audit_pic_full`. Executed quality probe SHA-256:
`0e6df612ac28a259dbffcd4e922b1849fb6c93ff204979420bdbc2c0b3523ac9`;
phase probe:
`d4d5a77a3266bf18044a8094cec5a401da3e9456950868c7468a5fcf70464e7f`.
Before adding the later render-arrival handoff audit, these exact probe versions
were preserved as `output/p292/executed_quality_compare_no_pic60.py` and
`executed_raw_phase_no_pic60.py`. The server `audit_pic_full` snapshot remains
unchanged; no result here is attributed to a later probe revision.

GPU 0 quality audit ran 23:38:16–23:38:49 UTC; phase audit ran
23:39:47–23:39:53 UTC on 2026-09-26. The strict full-intervention code gate and nine
probe CPU tests passed independently. The independent numeric/causal report gate
is closed with no remaining material blocker. The subsequent diagnostic overview
has completed all-frame visual QA and independent artifact review; see
[the visual report](no_pic60_visual_p292.md). It does not pass the failed shape/rest criteria.
