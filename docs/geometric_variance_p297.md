# P297: let temporal variance observe the saved positional path

Status: optional implementation under review; no quality improvement established.
P295 resolves an explicit window-end position jump and a smaller immediate
layer-relaxation response. P296 screens whether removing parts of that endpoint
filter preserves geometry before any such split can be selected. P297 keeps
the current shared endpoint map and changes a different observable.

The current w_kin_var penalizes population variance of the stored physical V.
Direct layer/bond position changes and the PIC endpoint jump are not all visible
to V. With geometric_variance, define y0=x0, yt=xt for t<T, yT=the exact shared
promoted endpoint, and Ut=(yt-y(t-1))/dt. Replace only the existing variance input
by mean_p mean_t ||Ut-mean_t(U)||^2. The same w_kin_var and density-unit multiplier
apply. The existing physical terminal and running kinetic, contact and continuity
terms retain their physical velocity inputs. The old momentum variance is no
longer penalized by this one term; record both variances explicitly.

The bridge returns an independently owned post-layer X=[x1,...,xT] and seeds every
actual trajectory position in the backward pass. X's last seed is merged with
the separate xT seed, just as the terminal V seed is merged with vT. The ordinary
and captured CUDA implementations have separate explicit APIs. Legacy outputs
and defaults remain unchanged; previous-position and full-sequence bridge modes
are mutually exclusive. Start state is constant. Q and the exact H^T pullback of
the promoted endpoint remain in the endpoint map.

Gradient evaluation, line search, warm-start checks, replay-noise calibration
and final replay merit all use the same observable. A reused accepted path must
equal the validated trajectory arrays before promotion. No x/v/C/F/Fp update,
extra pin admission, renderer edit or interpolation is introduced. Configuration
requires shared PIC, T>=2, finite positive w_kin_var, geometric_rest off and
endpoint-only KKT pin admission off. Defaults remain off.

This is a policy experiment, not a guarantee of rest. For a fixed endpoint,
Var(U)=sum_t||dy_t||^2/(T dt^2)-||yT-y0||^2/(T^2 dt^2), so an evenly distributed
path can reduce variance and constant drift has zero variance. Record every
phase's raw/saved RMS displacement, separate raw-final and remap movement, net
displacement and total path length. A smaller scalar alone cannot pass a motion
gate. At T20, a single endpoint jump has variance coefficient19/400 times its
squared geometric speed. With w_kin_var200 that is9.5, versus P294's remap weight5
before common unit conversion; unchanged nominal weight is not equal strength.

Validation sequence:

1. CPU derivative and ownership tests for both bridges, actual post-layer X,
   mixed control channels, pins, T1, simultaneous/repeated/missing seeds and
   stale contexts. CPU integration compares independently reconstructed archive
   rates and accepted/replay merits in legacy and density units.
2. A small hyde06 CUDA graph/FD gate on the fixed mixed60 source subset, using
   the original dx/dt, followed by a cap1 integration pair. These are correctness
   checks, not physical-quality evidence.
3. Before interpretation of an isolated mechanism solve, hold actual entering
   state, prepared targets/layer/plan/arrival data, initial controls and positive
   render lambda fixed in both arms. A copied argument list alone does not prove
   prepared references equal. Admit only an originally outer-accepted snapshot.
4. Any full adaptive-lambda A/B is instead a whole-policy comparison. Preserve
   identical source/configuration except this flag, original bunny300k/T20,
   dt1/240, dx0.3062907543956724wu, loss36^3, eight inner iterations, shared PIC,
   no subcell shift, corrected committed-state outer render merit, cap60 under
   animations300. Report actual acceptance/stopping, not just the cap. Match
   material IDs and report all phases so earlier movement or a larger remap
   cannot masquerade as rest. Guard/pin failure or loss of thin-region supply
   and target fit precludes promotion. A successful prefix cannot replace full
   trajectory, hold, gallery and per-frame rendered QA.

The automatic lambda balancer uses the changed physics gradient. Its response
is part of a whole-policy comparison and must not be described as fixed-lambda
causal evidence. No new rendering experiment is included here.
