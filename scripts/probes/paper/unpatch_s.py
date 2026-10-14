"""unpatch_s.py REF_REPO TARGET_REPO -- D145: undo patch_s.py in TARGET_REPO's scripts/pipeline_run.py: the kept-frames write
is replaced by REF_REPO's original full-archive block (np.savez of <out>_render_full_dt_iso_nn.npz with the J samples), so
that a run of TARGET_REPO writes the full archive again. Asserts each block exists exactly once."""
import sys
from pathlib import Path

ref = (Path(sys.argv[1]) / "scripts" / "pipeline_run.py").read_text()
p = Path(sys.argv[2]) / "scripts" / "pipeline_run.py"
s = p.read_text()
end = '    out["arms"][ARM] = {'
a0 = ref.index('    np.savez(f"{args.out}_{ARM}.npz"')
orig = ref[a0:ref.index(end, a0)]
b0 = s.index("    keep = sorted(set(list(range(0, int(dn), 12)) + [int(dn) - 1]))   # D145")
cut = s[b0:s.index(end, b0)]
assert s.count(cut) == 1 and ref.count(orig) == 1
p.write_text(s.replace(cut, orig))
print("restored the full-archive write in", p)
