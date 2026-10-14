"""patch_s.py REPO -- D145: REPO's scripts/pipeline_run.py writes the kept frames (every 12th simulated frame and the last
delivered one: <out>_frames12.npz with frames, raws, tgt, deliver_n, exactly tmp/frames12.py's cut) in place of the full
archive (<out>_render_full_dt_iso_nn.npz). Output only: the simulation, the JSON and every metric are computed before
it, unchanged. As D146's repo_d146h. Asserts the replaced block exists exactly once."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "scripts" / "pipeline_run.py"
s = p.read_text()
old = s[s.index('    np.savez(f"{args.out}_{ARM}.npz"'):s.index("    out[\"arms\"][ARM] = {")]
assert s.count('np.savez(f"{args.out}_{ARM}.npz"') == 1 and "Fg_commits=np.zeros((0, 0, 3, 3), np.float32))" in old, old
new = ('    keep = sorted(set(list(range(0, int(dn), 12)) + [int(dn) - 1]))   # D145: the kept frames only, no full archive\n'
       '    np.savez(f"{args.out}_frames12.npz", frames=np.stack([frames.x[k] for k in keep]), raws=np.asarray(keep),\n'
       '             tgt=tgt, deliver_n=np.int64(dn))\n')
p.write_text(s.replace(old, new))
print("patched", p)
