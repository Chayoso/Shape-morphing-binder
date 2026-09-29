# P328 prepared withdrawal CUDA receipt

Frozen commit4ab8510, hyde06 GPU0, checkout
`/data/relcfd/chayo/physmorph_v2/work/p303/code_prepared_withdrawal1`.
Launch mode `prepared-withdrawal-verify`, tag `p328_cuda1`; shared GPU launch lock
and50-second interval were preserved. Four actual CUDA tests passed, zero skips,
in2.17s. Both log and JUnit XML are included.

N27,T20,dt.002,dx.5,16^3 is a numerical fixture. It does not establish N300k
memory headroom, runtime, optimizer improvement or rest. No production loss,
default control policy or renderer changed. The CPU gate passed26 cases with
independent refutation; stream-preflight and post-step pinned V/C health findings
were closed before freezing this source.

`source_sha256.json` binds the actual source and receipt bytes. The log includes
both reduced-mode directional derivatives at.001/.0005, under unchanged
2% relative/5e-6 absolute tolerance and nonzero-signal gate. The other cases check
aligned streams, early mismatch rejection and autograd callback lifetime.
Independent receipt review verified the server originals, all eight manifest
entries, imported fixtures and four named XML cases; all four finite-difference
comparisons pass the unchanged gates. This audit introduced no new numerical run.
