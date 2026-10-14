# mkrepo_d148f.py SRC DST -- the D148 hook copy: SRC (a deploy of settled-base with --gap_split) + D140's read-only dump of
# windows 1-9 (the 0-1.5 s shell-against-bulk reading, tmp/d140/d140_dump.py, as repo_d146h); the run writes frames12 in
# place of the full archive (as repo_d146h). Nothing the run reads changes.
import pathlib
import shutil
import sys

B = pathlib.Path("/data/relcfd/chayo/physmorph_v2")
src, dst = B / sys.argv[1], B / sys.argv[2]
if dst.exists():
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
shutil.copy(B / "tmp/d140/d140_dump.py", dst / "physmorph/pipeline/run/d140_dump.py")
p = dst / "physmorph/pipeline/run/runner.py"
s = p.read_text()
old2 = '            write_term_dump(cfg.term_dump, a, x, stats.pop("term_grads"))\n'
assert s.count(old2) == 1
s = s.replace(old2, old2 + '        _d140.dump(a, x, x_start, tgt, cfg, len(frames) - 1)   # D140 probe (hook copy only)\n')
imp = "from ..window.telemetry import write_term_dump\n"
assert s.count(imp) == 1
s = s.replace(imp, imp + "from . import d140_dump as _d140                  # D140 probe (hook copy only)\n")
p.write_text(s)
q = dst / "scripts/pipeline_run.py"
s = q.read_text()
i = s.index('    np.savez(f"{args.out}_{ARM}.npz"')
end_tok = 'Fg_commits=np.zeros((0, 0, 3, 3), np.float32))\n'
j = s.index(end_tok, i) + len(end_tok)
s = (s[:i] + '    keep = sorted(set(list(range(0, int(dn), 12)) + [int(dn) - 1]))   # probe: frames12 only, no archive\n'
     '    np.savez(f"{args.out}_frames12.npz", frames=np.stack([frames.x[k] for k in keep]), raws=np.asarray(keep),\n'
     '             tgt=tgt, deliver_n=np.int64(dn))\n' + s[j:])
q.write_text(s)
print(f"{dst} ready")
