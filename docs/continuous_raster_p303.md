# P303: continuous raster support and a current-core raw-endpoint comparison

Status: experimental. The raster cutoff mechanism has been isolated and an
opt-in CUDA repair passes the recorded operator tests. Physical holes, free
material rest and high-resolution appearance are **not solved**. No production
defaults or published render have changed.

## Raster mechanism and implementation

The loaded diff_gauss discards a Gaussian/pixel contribution at alpha<1/255.
That is a value discontinuity. Its backward differentiates the current active
set, while a finite control update can move pixels across the cutoff. A dense
independent Torch float64 oracle reproduces P302's 48-splat, 256x144, opacity.25
joint mean/covariance discrepancy: CUDA refreshed-mask FD differs from AD by
10.154%,8.451%,10.289% at steps .01,.003,.001; dense hard-cutoff differences are
10.153%,8.442%,10.300%. Freezing the mask makes the dense difference converge to
2.35e-5,2.09e-6,2.33e-7 relative. This isolates that toy's cutoff effect; it is
not evidence that every discrete builder operation has a smooth derivative.

The isolated `physmorph_diff_gauss` build uses
`alpha = min(.99, max(opacity*exp(power)-1/255, 0))` and skips only zero alpha.
It is continuous but not continuously differentiable at the support boundary.
It changes coverage throughout the support, so training and export explicitly
select the same backend. The source dependency and original `diff_gauss` are
untouched. Projected support bounds cover the full positive support, including
fractional tile boundaries. Saturated alpha gets a zero derivative, and the
clipped-FoV covariance depth derivative uses the correct chain rule.

No transmittance threshold terminates the new forward. A float64 log product is
saved in the image scratch buffer and reversed in log space, avoiding both
`1-alpha_image` cancellation and a permanently zero backward recurrence through
opaque stacks. The inherited CUDA extension uses stream0. An explicit GPU-event
adapter orders the caller and default stream around Function.apply, clones input
buffers into Torch-owned storage and records allocation consumers. It joins in
`finally`; the extension retains a fail-closed stream/device guard. This adapter
adds no host synchronization. The inherited extension still reads a bin count on
the host; this is not a host-free runtime claim.

`prepare_continuous_raster.py` verifies the complete1550-file input-tree digest,
checks the copied tree, applies exact-count edits, and writes a build receipt.
`build_continuous_raster.sh` builds only under the project's `/data/.../deps`.
The operator report binds the loaded Python module, binary, receipt and helpers.
The initial legacy discriminator predates those bindings; its unmodified JSON
is retained and must not be represented as having the later provenance fields.
Its `mask_flips` field means hypothetical live threshold flips even for the
frozen-mask oracle; the later probe labels applied/live masks separately.

## Operator evidence

The continuous CUDA 48-splat joint test has refreshed FD differences
0.1994%,0.03187%,0.00570% at the same three steps. Fractional tile sweeps,
8/64/256 colored overlapping splats, opacity saturation and off-axis covariance
clamping have maximum dense forward error5.58e-7 and maximum dense AD relative
error4.94e-6. Side-stream/default-stream images match exactly; gradient maximum
difference is2.05e-8. This is an operator gate, not a rendered quality gate.

The original300k source, one-native-spacing synthetic target translation,
four cameras, 456x256 coarse and full3840x2160 detail rasters with256-square
patches give y-translation FD differences0.0855%,0.000404%,0.0102% at
.1/.03/.01 source spacings. The older legacy test was approximately6%.
The continuous full-cloud gradient is finite, norm.159856; peak Torch allocation
is4.102GB and the complete operator script takes4.21s on hyde06. There is no MPM
tape in that memory figure. This direction does not certify arbitrary changes
of KNN/support/donor sets or visibility ordering.
The original live report predates embedded source/input hashes. Its separate
`live_continuous1_execution_receipt.json` binds the retained code5 snapshot,
launch files, input archive and actually loaded backend receipt after the run;
it does not retroactively add fields to that original report. The actual
`continuous_raster2_build.json` is retained with the evidence.

CLI: `--surface_gs_raster continuous` with the opt-in shared surface loss;
export: `--surface-common --raster-backend continuous`. Both retain `legacy`
as default. Shared GS still requires the existing shared-PIC endpoint contract.

## Current-core physical comparison

All three completed physical runs use the same numerical source hash recorded
in `evidence/p303/provenance.json`, original N300000 inputs, T20, dt1/240,
dx.3062907543956724wu, loss36^3, an eight-iteration inner budget and cap24.
P300's constitutive adjoint repair is active; `shift_sub=False`; GS is off.
Baseline→raw changes exactly `commit_pic` and `commit_pic_objective` true→false.
This changes the optimized policy, not merely final postprocessing. P292's older
PIC-off trial retained subgrid shifting and a different adjoint, so this was a
necessary new comparison. Its contrary coverage result remains consistent with
P295/P301's direct geometric evidence.

Both deliver24 accepted windows/481 raw states, with zero recorded repairs or
inversions. Independent raw checks verify finite in-bounds archived positions,
compact/raw endpoint equality and zero raw-mode commit_other/PIC/shift motion.
This does not establish stored-v/geometric-velocity consistency: layer u and
relaxation remain direct position updates inside the rollout.

| At common commit24 | Shared PIC | Raw endpoint |
|---|---:|---:|
| Independent binary silhouette IoU |.968871|.963604|
| Raw Chamfer, wu |.058829|.059339|
| Fixed upper-target coverage |.965060|.936455|
| Moving upper-region density |.900787|1.132840|
| Tip-ball particles, target89 |43|80|
| Fixed source-upper cohort density |.703926|.955490|
| Pinned fraction |.877547|.906020|

The same3352 IDs free in BOTH arms at commits14–24 have saved-state step RMS
.103175→.048158 source spacings and sampled reversal fraction4.922%→.103%.
Phase20 includes the final rollout step plus commit corrections in the baseline;
the raw arm has zero PIC/other commit correction. These are not unprojected
physical-step RMS measurements.
The last phase's total path share falls20.95%→5.31%; reversals into that phase
fall49.38%→.0418%. This is substantially smoother observed motion but is not rest
or a periodic-vibration measurement. All observed admitted pins remain exactly
still (259606/270127 IDs with subsequent observations); new final admissions have
no subsequent physical observation. The raw policy fails the joint fit/coverage
gate and is not promoted. Coverage already differs at W2 before either arm has
pins, although the late pin population may still restrict supply.

## Rendering influence

Raw endpoint on/off changes only `lambda_auto` .5→0, with the same discretization,
inputs, source and cap24. Rendering on/off gives IoU.963604/.955043,
Chamfer.059339/.059455wu, global target coverage.993163/.991650 and tip80/59.
Density is1.132840/1.225856, so a density gain alone is not better coverage.
These are full policy effects, including later controls and pin decisions.

Across accepted windows, nominal render direction share medians are .4105 for
shared PIC and .4372 for raw. Raw per-channel medians are body.2896,
stress.4835 and surface-u.7046; median adaptive lambda is.04728. These are gradient
norm bookkeeping, not percentages of physical displacement. Standard run
`*.render_influence.json/.md` additionally preserve actual accepted update
norms and image-loss changes. The existing coarse CIC/PBR channel contributes to
shape here; it does not directly supervise the exported4K Gaussian appearance.

## Pin admission ablation: not a joint solution

The raw control repeat and pin-disabled policy execute all 24 windows at the
same N300k/T20/dt1/240/dx.3062907544/loss36^3, eight-iteration budget. Delivery
selection retains only W23 in the pin-disabled arm. Comparison therefore uses
the common **delivery-retained W23**; this is not a claim of only 23 successful
physical commits. All recorded state guards are zero.

| At common delivered W23 | Raw repeat, pins on | Pins off |
|---|---:|---:|
| Binary silhouette IoU | .963819 | .957751 |
| Chamfer, wu | .059408 | .059144 |
| Fixed upper-target coverage | .929924 | .931100 |
| Tip-ball particles, target 89 | 70 | 95 |
| Fixed source-upper density | .965621 | .999721 |
| Matched-free saved-step RMS, sp | .046462 | .051070 |
| Matched-free committed-step RMS, sp | .884003 | .978683 |

The motion row uses the same 4,952 IDs free in both arms over W13--23, selected
at that common endpoint; it is descriptive rather than a pre-treatment cohort.
Their committed-step reversal fraction rises 4.71% to 11.24%, while saved-step
reversals fall .1295% to .0631%. Neither measure establishes periodic vibration.
Removing pins does not simultaneously improve fit and material rest.

The first two windows admit no pins. At W1, pin-off/original-control endpoint
RMS is .00012695 wu, versus .00001597 wu for one same-policy repeat; at W2 those
are .00066445 and .00059660 wu. One repeat is not a noise distribution. Do not
attribute every later difference solely to pin admission or conclude that pins
caused the earlier PIC/raw coverage gap. Pin policy also contains assimilation
and control suppression, so this is not an arrival-radius-only intervention.

Pin-disabled export initially encoded optional `None` pin fields as object
arrays. A new archive replaces only those two fields with typed false/-1 arrays;
all seven other member payloads and original metadata remain byte-identical.
No pickle is loaded, and zero checked pin IDs is a vacuous pin-motion test.
The receipt and exact executed normalizer are retained. The completed prefix
report originally bound metadata only; its separate execution receipt binds
the retained archives and executed source after the run. Later source guards
must not be represented as having executed in that older report.

## Continuous GS loss: operator repair does not imply shape improvement

Matched shared-PIC, no-shift runs use the continuous backend, N300k/T20,
dt1/240, dx.3062907544, loss36^3, **one window and two inner iterations**.
GS weight 0/1 and a weight-0 repeat all accept two inner steps and one outer
commit, with zero guards. Weight 1 improves detail coverage/edge image losses
about 7.82%/4.83%, but global GS coverage worsens about 19.5% and total GS loss
worsens 1.73%. Raw IoU falls .731116 to .726179, Chamfer rises .167825 to
.170180 wu. Adaptive lambda changes .327158 to .013993. Endpoint displacement
between policies is .026620 wu RMS, versus .00001894 wu for the single repeat.
This exceeds the observed repeat discrepancy but does not establish a noise
distribution. No GS-weight or continuous-backend quality promotion follows.

## Continuing gates

The next read-only test holds the observed trajectory, current lambda and
current physical/cleanup terms fixed while swapping the previous/current
prepared density and CIC/PBR references. It tests whether target refresh
rewards residual movement before selecting a new rest policy. No full-gallery,
all-frame visual, watertightness, persistent individual-rest or high-resolution
artifact gate has passed in P303 yet.

Evidence JSONs are in `docs/evidence/p303`; full experimental raw archives remain
under hyde06 `work/p303`, outside the retained before/corrected deliverables.
