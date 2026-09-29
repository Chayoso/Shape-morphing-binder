# P330 complete head-merit CUDA gate

Frozen cfe940b1a45722dc08f68485be8b4974c35f4e75 under
`work/p303/code_checkpoint_merit1`, hyde06 GPU0, `p330_cuda1`.
Both named captured-joint cases pass, zero skips,8.06s. Discretization:
N160,T3,dt1/240,dx1wu,grid32^3,loss12^3,iters2. Each reduced body mode uses
the same complete head objective, two-radius finite differences and callback
expiration check. This is a capability fixture, not P329's N300k checkpoint.

`p330_cuda1.log` retains numerical witnesses; `p330_cuda1.xml` identifies both
executed cases. `p330_source_sha256.json` records the three reviewed file hashes
before launch and after, and83 post-run source hashes. Other source files were
not individually hashed before this test; do not broaden that scope. The receipt
script is copied as `p330_source_receipt.py.txt`.

The source/CPU launch gate passed independent review;39 CPU regression cases
pass, including11 new cases independently rerun by two reviewers. The small
docstring correction after review changed no numerical logic. Independent actual
CUDA receipt audit passes: both named cases and all four finite-difference
comparisons satisfy the unchanged gates. Relative errors are.004877%,.002609%,
.519659%,.653242%, with both AD magnitudes above50e-6. All three reviewed
before/after hashes, log/XML/receipt-script hashes and imported fixture bytes match.

The83 post-run inventory contains82 files matching local working bytes and the
frozen commit content (19 differ from Git blobs only by worktree newlines), plus
one unreferenced remote-only legacy `physmorph/volumetric.py`, SHA256
`aca76c8ae15ef14a7cd9e9d0c5a314cfeb4b3c8edb6132d372588956a5eb5c84`.
The reviewed optimizer/runner/prepared-reference paths import
`physmorph.losses.volumetric` instead. This qualification is preserved in the
receipt inventory; no claim that all83 are tracked commit files is made.
After the audit the canonical deployed copy of that unused module was archived
with byte/hash verification under `maintenance/archived_sources`, then removed
from the canonical repo. `canonical_legacy_archive.json` records that exact file;
the cleanup script is retained here. All frozen test/producer trees, including
their inventoried legacy module, remain unchanged. Future canonical copies will
not inherit it. No MPM or rendering code was changed by this cleanup.
No default loss, accepted state or rendered artifact
is changed, and no rest/hole/4K quality gate is implied.
