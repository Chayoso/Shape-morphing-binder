# P313: locate support changes in saved P312 trajectories

Status: read-only diagnostic implemented, four focused CPU tests independently
pass and implementation review is closed. No new forward simulation or policy.
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
