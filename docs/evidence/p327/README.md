# P327 evidence

`p327_cuda1.log` and `.xml` retain the8 actual CUDA passes from frozen6a525c3:
six P326 joint derivative/fragment cases andtwo P327 observer cases, zero skips.

`production_summary.json` reduces the completed source-bound JSON reports only.
It contains the exact SHA256 of every input and of `summarize_production.py`;
the script performs no geometry or simulation. Its `ROOT` is the local archival
download directory `C:/dev/physmorph_runtime/p303/p327_evidence`. Detailed large
reports are retained on hyde06 under `/data/relcfd/chayo/physmorph_v2/work/p303`:

- `p327_legacy1` and `p327_retained1`: full raw traces, configs, inputs, archive
  bindings, fragment protocol/activity and rendering-influence sidecars.
- `p327_quality1.json` and `p327_phase1.json`: bound same-code intervention and
  common accepted physical-clock comparison.
- `p327_{legacy,retained}_motion1/result.json` and `/particles.npz`.
- `p327_{legacy,retained}_shape1/result.json` and bit-packed mask chunks.

The production checkout is `code_fragment_adjoint1` at6a525c3. The analysis
checkout is `code_fragment_analysis1`: the same producer bytes with only the
pair comparator and phase script overlaid from64e97a3. No runtime wrapper is
identified from equal producer hashes alone.

The pair is descriptive: byte-level path differences start before recorded
time-varying fragment exposure. Both fixed-source and outcome-selected free
cohorts are preserved; own endpoints use different clocks. Projected openings
remain in both arms. Rendering norm shares are not causal movement shares.
See `docs/fragment_adjoint_p327.md` for interpretation and independent audit scope.

`independent_audit_index.json` binds the included source/archive, motion, shape
and pair-metadata receipts; CUDA execution receipts/logs are also retained.
The unchanged bounded P316/P317/P319 auditor scripts are identified by exact
hash in their protocols. No audit reruns MPM or a renderer.
`compare_prefix_bytes.py` and its first12-attempt report establish bytewise
nonidentity only; absence of early temporal-mask exposure comes separately from
the two bound activity reports. They do not measure numerical difference size.
