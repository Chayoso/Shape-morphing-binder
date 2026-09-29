# P334 independent receipt and result audit

Scoped gate: **PASS**, with no remaining material finding. This audit read the archived CUDA result; it did not launch a GPU or repeat the numerical experiment.

The frozen commit is `c4a7c82f66cc6c5dd883f0ea27aa54165145f6f8`. All 92 source bindings match the current local files byte for byte. Against Git, 73 are byte-exact and 19 differ only by CRLF line endings; there is no remaining content mismatch. All five locally available receipt-bound artifacts match their size and SHA256. The XML and receipt agree exactly on nine passing cases, zero failures/errors/skips, exit 0 and 5.754 seconds.

Recalculation of the 12 printed FD rows retains the registered `max(0.02*abs(FD), 5e-6)` allowance. All full derivatives pass. At epsilon 5e-4 the missing-Fp surface-u derivative fails: full -.0183089683, FD -.0182145925, omitted -.0104908700, allowance .0003642919. The missing-C body derivative fails: full -.00711814382, FD -.00713741616, omitted -.00612606205, allowance .00014274832. The other omitted-channel rows do not fail; the test requires at least one observable witness per removed path.

`tests/test_post_assimilation_adjoint_cuda.py:45` requires the four Warp and two Torch graphs. Lines 156?214 evaluate each omission on the same primal, forbid eager head/coast/boundary fallback, and intercept the live captured boundary covector only. Changed controls, owned outputs/policies, seeds, lifetime and stream cases complete the nine-case scope. These tests do not establish the derivative of discrete admission or changing successor preparation.

The local memory JSON equals the receipt, and its 14 samples give process/device maxima of 1300/1324 MiB. These are sampled maxima. The 77,414,440,492-byte project size is the collector's measurement, not a second remote measurement by this reviewer. The preserved original launcher `run_p334_v1.sh` matches its receipt byte count (1155) and SHA256 (`cc2c99db89d6308212625cebd6a8f7cb74df344f33aaeafaa847679c5de0aee9`). Installed shared-library binaries were not hashed.

The result document, P333/plan cross-references, experiment entry and AGENTS note retain the N27/T20/dt .002/dx .5/grid16^3 operator scope. They make no physical-rest, hole-removal, candidate-adoption or visual-quality claim. Actual prepared production handoff and large-run forward/merit parity remain separate gates.

Machine-readable details: `independent_audit.json`, SHA256 `8bb6f1efa84b5b736df6b19c308466eea13b1aa0d4e8523b6dbd7a265a20904e`. Original receipt SHA256: `7f181382303af7647a184c8513c97d0e670b93dbb05c379e65018eff18206057`.
