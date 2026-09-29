# P313: locate support changes in saved P312 trajectories

Status: archive-only CUDA analysis completed (`coverage_paths1`, frozen `42c01a0`).
Four focused CPU tests independently pass; independent brute-distance CUDA
verification and result review pass. No new forward simulation or policy.
Use only `silhouette_repair1`'s saved original baseline triplet and its actual
shared `terminal05_origin_1.npz`. Do not analyze the unsaved h9/c2 candidate by
rerunning it or claim the origin's target IDs are that candidate's IDs.

Discretisation: W20,N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,budget8,
raw/no-PIC/no-shift. Bind the result/protocol, saved arrays and original
source/target to their existing checksums before and after analysis. Bind the
current analysis code and unchanged metric/KDTree source. Decode numeric arrays
from the live archive without constructing FrozenBodyWindow or an adjoint.
All distance, set, cohort and trajectory calculations run on hyde06 CUDA;
archive decoding/hashing and report serialization are declared host I/O.

Recompute the original target spacing using the existing exact KDTree backend:
median second-neighbor distance. Keep the existing inclusive cutoff2*spacing
and upper-target predicate y>2.3. Recompute target coverage on all four saved
endpoints and require closure with their recorded raw fractions. Do not modify
the original gates. Verify the source/target arrays match the captured inputs.

For every target ID, save its four endpoint NN distances and supplying material
IDs, covered bits and signed margins (cutoff minus distance), in wu and native
source-spacing units. Compare the terminal origin with each baseline: separate
lost and gained IDs from the net count. Report the AND-covered, AND-uncovered,
and ambiguous-across-baselines masks. Baseline repeat envelopes are descriptive
observations, not new acceptance tolerances. Include all changed IDs and all
baseline-ambiguous IDs in subsequent path analysis, without choosing a subset
based on the sign or size of the difference.

For each selected target, save the four nearest material IDs at each of the four
endpoints. Freeze the union of these IDs when tracking their X/V trajectories
through all20 saved substeps plus x0. Report their start-pinned,
start-arrived-free or remaining-free membership. Save the target-to-material
distances along those fixed paths, and separately recompute actual phasewise
nearest IDs/distances using all300k particles at every saved phase. A change
of nearest identity is not material motion. The endpoint union is a bounded
witness set, not every contributor: retain phase-nearest IDs outside each
target's endpoint set and report their occurrence explicitly. Distinguish this
from membership in the global union whose full paths are saved; another target's
supplier does not belong to this target's endpoint set. Save per-phase occupancy counts
inside the unchanged coverage cutoff. NN ties are not uniqueness evidence.

Write numeric NPZ sidecars for the complete endpoint table and selected paths,
plus JSON listing IDs, lost/gained/stable/ambiguous counts, margins, onset phase
and cohort membership. Obtain common x0/pins from the hash-bound live observation;
the four trajectory NPZs do not independently store them. Require every saved
pinned position to equal common x0, bind the cohort masks, and require
finite expected-shape arrays. Preserve the saved physical V and separately
compute geometric increments; do not equate them or invent v0.

This test can localize where terminal braking changes support and whether loss
appears early or late in the saved window. It cannot establish that continued
movement is necessary, that a tiny margin is meaningless, that all visual holes
are explained, or that a repair persists under coupled continuation. The shared
origin already fails coverage; no promotion is possible from this analysis.
Report P312 render influence as inherited evidence, with no fresh render-loss
evaluation or claimed rendering intervention. Independent refutation of the
tool and results is required before using the diagnosis to choose a repair.

## Observed support changes

At the stated W20 discretisation, the saved endpoint coverage counts reproduce
all four original records exactly:297701/300000 in every baseline,297699 in the
terminal origin; upper coverage is14241/15312 throughout. The three baseline
coverage bitsets are identical (no ambiguous target ID). The fixed cutoff is
.0698616982373597wu, twice target spacing.03493084911867985wu; source native
spacing is.03498853660707278wu. Four targets lose coverage and two gain it.

| Target ID | Endpoint change | Original margin (wu) | Terminal margin (wu) | Phases with less coverage than original |
|---|---|---:|---:|---|
|24591|lost|+1.32629e-4|-5.44419e-5|20|
|241029|lost|+2.93170e-4|-4.54429e-4|20|
|282449|lost|+1.03281e-4|-5.44937e-5|20|
|287243|lost|+2.96007e-5|-4.99621e-4|8,20|
|58264|gained|-5.36804e-5|+1.55122e-5|11|
|194769|gained|-1.35695e-4|+1.38848e-5|none|

Margins use original repeat0 and cutoff minus actual nearest distance. A phase
comparison concerns relative coverage, not uninterrupted supply loss. Target
24591 is covered by the original only at its final phase: this is a missed final
entry. At287243 both trajectories are uncovered at9..19. Targets241029 and282449
retain coverage longer earlier in the modified trajectory, despite losing it
at the endpoint. Target58264's endpoint gain also coexists with an earlier
relative loss. Thus the endpoint count alone cannot identify a persistent hole
or decide which motion should stop. Tiny margins do not invalidate the failure.

Every selected target has the same endpoint-nearest material ID in all four
arrays. All six endpoint-nearest suppliers are start-arrived-free. The full
endpoint4-NN witness union contains24 material IDs:22 start-arrived-free and
2 pinned farther witnesses. No phase-nearest ID lies outside its own target's
endpoint witness set, or the global union, in these observed trajectories.
These cohorts are coarse transport labels, not convergence/rest certificates.

P313 performs no new rendering. It inherits P312's original lambda.01655326075;
at this terminal05 origin, prepared silhouette decreases3.66708e-8, PBR1.21596e-7,
combined render1.58325e-7 and weighted render2.62079e-9 versus original repeat0.
This improvement accompanies the raw coverage loss. It cannot establish rendering
causality or4K visual quality; the prefix uses18 views at64 pixels.

No policy is adopted. `support_repair_p314.md` describes a separate opt-in
diagnostic to test whether fixed material support constraints can prevent an
identified endpoint deficit while reducing running motion through actual MPM.

Independent `coverage_brute1` checks all84 saved states against all300k particles
for the six selected targets (504 target/phase observations), using direct
Torch FP64 distances on CUDA instead of KDTree. Coverage bits, occupancy and
nearest IDs are exact. Maximum nearest/4NN distance difference is1.38778e-17wu;
fixed-witness distances/geometric velocities match exactly, and saved witness
X/V and all original pinned trajectories are bitwise exact. All input/code
checksums and file identities match before/after. The audit reports2.538s elapsed
inside its main function, including archive decode/hash checks, and450.3MB peak
Torch allocation. Imports and launcher overhead are outside that timing; this
is not a full-pipeline performance benchmark. Evidence and the independent audit
source/launcher are retained in `evidence/p313/`. The full endpoint table is
retained under the checksum-bound server/local runtime paths in its protocol.
