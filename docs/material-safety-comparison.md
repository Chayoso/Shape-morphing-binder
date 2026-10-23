# Local LLM material pipeline and numerical safety

Based on `settled-base` commit `3219c9ceec434f22165ed13c64dd24804bf9e296`.
Native solver, optimizer, loss, material model, evaluation gates and renderer are unchanged.
All material inference, validity checks, output sampling and GIF assembly are external.
No A/B/C modifications, extra force or reference trajectory is enabled.

## Text to GIF

Use the project's working NVIDIA CUDA simulation/render environment (see README),
including the existing Gaussian rasterizer and FFmpeg with NVENC. Add the language
and display dependencies without replacing a working CUDA stack:

```bash
python -m pip install -e ".[language,dev]"
python scripts/download_language_model.py
python -B scripts/text_morph.py "A soft rubber bunny morphs into a dragon." --out output/rubber
```

The download command is an explicit one-time download of the pinned Qwen snapshot
(approximately 8 GB). Weights are never committed. Inference then uses local files only.
If the weights already exist elsewhere, skip downloading and pass `--model-dir /path/to/model`
to `text_morph.py`. No API key or hosted inference service is needed.

The main entry defaults to seed97/N40000 and inherits all native optimization settings.
Supported existing shapes: bunny/rabbit, dragon, sphere/ball, armadillo and cow.
English descriptions are required. Default `--material-mode infer` uses two Qwen calls:
literal shape/material evidence extraction, then E/nu range estimation. This mode has
no runtime material-table lookup. Explicit labeled E (Pa/kPa/MPa/GPa) and nu take precedence;
ambiguous input, unsupported shapes, invalid JSON or unphysical parameters stop before simulation.
`--material-mode preset` explicitly selects a documented preset through the same language entry.

LLM values are uncalibrated model estimates, not measurements. Raw ranges, assumptions,
limitations, model/revision, prompt/responses and selection/conversion rules remain in
`<out>_request.json`. E is the geometric midpoint of its range; nu the arithmetic midpoint.
All inferred materials use the same conversion `E_solver = E_Pa / 89285.71428571429`;
nu is dimensionless and unchanged. This is a common solver-unit convention, not A's
material-dependent amplification of dFc. Preset sources/rules are recorded separately.
It does not establish real-world scale, density, SI time or finite-strain rubber fidelity.

Language, simulation and rendering use separate GPU processes so model memory is released
before simulation. Windows entry scripts load torch before CuPy to select the working CUDA DLLs.
Safety is always enabled. Defaults use native dt; `--substeps 2` explicitly
selects the disclosed steel refinement below, rather than silently changing the material law.
`--res` changes the final display resolution only (default320), not the optimization loss.

Final outputs are `<out>.gif`, `<out>.md` (metrics, gates, safety counts, configuration and
provenance) and `<out>_request.json`. Temporary trajectories, state/Adam dumps, MP4/PNGs
and worker logs are deleted after that run, including on failure. A scalar error/log tail
is retained in the MD. Existing outputs are never overwritten. A run that completes with
failed native gates is clearly labeled and returns a nonzero status; final gates do not
certify target fit. Invalid trial rejection cannot guarantee success for every material.

The GIF reuses the original native shader/camera. It samples the delivered trajectory at
0.2 solver-time intervals, preserving the exact final endpoint, and plays it at2x.
Initial/final display pauses add no simulated time. Labels show material, E/nu, final IoU
and the solver clock. Rejected branches and display-only terminal frames are excluded.

## Reuse the safety wrapper

```python
from scripts.experiments.material_state_safety import state_context

with state_context(lambda event: print(event)) as safety:
    # Your existing pipeline call, or scripts.pipeline_run.main().
    result = run_pipeline(...)
```

It checks every recorded x/v/F state and the native orientation/volume criterion before
surface-grid or loss construction. Invalid trials receive infinite merit and use native
backtracking/rollback. Invalid warm starts can fall back to ZERO. An invalid taped forward
is blocked before loss/AD; if no valid start remains, the window returns a null commit,
clears stale transport/adjoint caches and leaves promotion/rollback to the native runner.
Counters and the first invalid-state locations are emitted through the supplied callback.
`diagnostic=True` is an optional stop-on-nonfinite localization mode, not a normal run mode.

Use one context per process, without nesting or concurrent threads: it temporarily patches
the existing Python entry points. It does not alter material laws, gates, merit thresholds
or gradients, and does not cure unstable physics or native CUDA/driver process aborts.
The comparison supervisor uses `CUDA_MODULE_LOADING=EAGER`, as in the validated Windows runs.

## Run the comparison

From the repository root, using the project's existing CUDA environment:

```bash
python -B scripts/experiments/run_material_comparison.py run --out output/material_comparison
```

`--out` must be a new directory. Optional `--python /path/to/gpu/python` selects the worker
interpreter when the CPU supervisor uses a different Python installation. The native
renderer still requires its existing rasterizer and FFmpeg/NVENC setup; Pillow is used
for the composite. This optional fixed comparison needs no model inference or old ABC module; the
text-to-GIF entry above is the complete LLM pipeline.

Inputs are bundled in `scripts/experiments/material_comparison.json`: soft/hard rubber,
PMMA, wood, concrete, aluminum alloy and steel. Raw ranges/units, source prompts,
estimator revisions/hashes, assumptions and limitations accompany each selected value.
E uses the geometric range midpoint; nu uses the arithmetic midpoint. These are frozen
LLM estimates, not measured material constants. Every E is converted by the same stress unit:
`E_solver = E_Pa / 89285.71428571429`, with nu unchanged. The unit was anchored by
`12.5e9 Pa / 140000 solver units`; it preserves modulus ratios and is a simulation convention,
not a physical validation of object scale, density, yield or SI elapsed time.

All cases use bunny-to-dragon, seed97/N40000 and native CLI defaults. Safety stays enabled.
Steel alone uses `refine_time(2)`: half dt, twice the integration steps, the original20
optimized dFc bins split over microsteps, and the same total window duration. This fixes
the observed coarse-step startup divergence without adding a force or control freedom.
It is a disclosed discretization change; equal finite-step stress histories are not claimed.

The GIF uses actual step*dt timestamps for every material, a common0.2 solver-time sample
interval, and no per-material time stretching. Rejected branches, best-truncated frames and
terminal display holds add no simulated time. Short runs hold their delivered endpoint.
Labels contain only one shared clock, material, E/nu and final IoU. Initial/final display
pauses are presentation holds. The existing native shader/camera are reused.

One attempt per material; no retries or sweeps. Each disposable simulation/render job is
removed immediately after its panel images are loaded into supervisor RAM. Final outputs
are `results.md` (scalar configuration, metrics, gates, safety and provenance) and
`materials.gif`. Exceptions stop the batch and retain the scalar error in `results.md`.
No dense controls, Adam caches or full trajectory archives are retained.

## Existing evidence and preparation checks

The safety-wrapper lineage was exercised in the completed30-run baseline/A/B/C/AC matrix,
including fixes during that workflow. The helper packaged here is byte-identical to the
version used for the subsequent seven-material comparison and repaired steel display.
These historical executions are not new tests of this portable packaging.
The steel row below is the repaired run shown in the GIF.

| Material | IoU | Chamfer |
|---|---:|---:|
| Soft rubber |0.641068|0.371094|
| Hard rubber |0.652073|0.362763|
| PMMA |0.973913|0.091181|
| Wood |0.974661|0.090814|
| Concrete |0.972841|0.093320|
| Aluminum alloy |0.971772|0.095383|
| Steel, refined |0.969593|0.096634|

All seven passed the existing final safety gates; this does not certify target fit or real-material fidelity.
GPU nondeterminism means these numbers are reference readings, not bitwise reproduction promises.

Preparation checks: fixed parameter/unit loading, shared-time endpoint handling, exclusion
of rejected/display-only frames, invalid trials bypassing geometry/loss, and invalid AD
forwards producing safe null-window/cache recovery. These use CPU fixtures only:

```bash
python -B tests/test_material_safety_comparison.py
```

No ABC variants, independent/reference actuation, local agent instructions, model weights,
environment caches, historical dense run dumps or GIF binaries are committed.

Text orchestration checks (request integrity, safe-worker routing, GIF generation and
cleanup on success/failure):

```bash
python -B tests/test_text_gif_pipeline.py
```

Packaging validation: nine focused CPU checks passed. A full CUDA text-to-GIF run
is not claimed as validated: an initial Windows DLL-order failure was isolated and
corrected; the subsequent simulation was stopped at the user's request.
