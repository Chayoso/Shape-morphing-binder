# P329 production withdrawal receipts

Producer19e725ec97e8229cafc8a1680bdebc23d45995fe, frozen under
`/data/relcfd/chayo/physmorph_v2/work/p303/code_production_withdrawal1`.
Run `work/p303/p329_live1`; N300k,T20,dt1/240,dx.3062907543956724wu,
loss36^3,iters8,18 render views64px,GSoff. Read-only W20/inner8 observation;
no coefficient proposal, adoption, rest claim or new rendered deliverable.

- `protocol.json` binds recipe, physical parameters, source/target and source
  bytes before/after. Only stop_after_windows changes to20.
- `result.json` contains final callback/outer disposition and complete original
  history; `checkpoint.json` is the earlier measurement receipt, before the
  original outer return/commit. Do not substitute it for final outer acceptance.
- Full accepted/private/joint/prepared-owner/gradient NPZ witnesses remain on
  hyde06 under the run folder. Their sizes and SHA256 are in the checkpoint and
  final result. They are deliberately not duplicated into this repository.
- `p329_live1.log`, `run.render_influence.{json,md}` retain original execution
  and nominal rendering-direction telemetry, not a render-off causal estimate.
- `p329_live1.memory.json`, `sample_device_memory.py.txt` and
  `run_p329_live.sh.txt` bind the external PID/starttime observer. Sampled
  device/process memory includes allocations outside Torch, but can miss peaks.
- `p329_raw_coast1.{json,log}`, `p329_archive_quality.py.txt` and
  `run_p329_quality.sh.txt` report archive-only raw geometry for21 coast states,
  fixed source IDs and24views at128/256px. Numerical operators run on CUDA.
  No renderer, loss, model construction or MPM replay is used by this observer.

Implementation gates:24 CPU cases passed, independently refuted, including
deliberate owned-C corruption that preserves failed evidence without changing
ordinary pipeline frames. The original source run's capability gate passes.
Independent actual-output/provenance and bounded CUDA archive audit is pending;
this status will be replaced only after its receipt arrives.
