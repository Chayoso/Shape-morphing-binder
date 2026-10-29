# mkrepo_d148rx.py -- repo_d148rx: repo_d148rf (refreeze-prep d2f779d + D140's read-only dump of windows 1-9 + frames12)
# + D146's read-only per-particle hook v2 (dF_c's rms and cap share, u and its gate, each term's gradient at the end,
# snapshots of every window from $D146_A0 on), here only up to window $D148_A1: the dump run RWX of D148e's
# pre-registration (16:16 CDT). The gate runs never use this copy.
import pathlib
import shutil
import sys

B = pathlib.Path("/data/relcfd/chayo/physmorph_v2")
src, dst = B / "repo_d148rf", B / "repo_d148rx"
if dst.exists():
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
shutil.copy(B / "tmp/d146/d146_dump_v2.py", dst / "physmorph/pipeline/run/d146_dump.py")
p = dst / "physmorph/pipeline/run/runner.py"
s = p.read_text()
old = '        if cfg.term_dump and stats.get("term_grads") is not None:\n'
new = ('        if a + 1 <= int(os.environ.get("D148_A1", "999")):   # D146 probe up to window D148_A1 (repo_d148rx only)\n'
       '            _d146.dump(a, x, x_start, commit, res, F, Fp_pre, cfg, len(frames) - 1)\n' + old)
assert s.count(old) == 1
s = s.replace(old, new)
imp = "from ..window.telemetry import write_term_dump\n"
assert s.count(imp) == 1
s = s.replace(imp, imp + "from . import d146_dump as _d146                  # D146 probe (repo_d148rx only)\n")
if "\nimport os\n" not in s:
    s = s.replace("from __future__ import annotations\n", "from __future__ import annotations\n\nimport os\n", 1)
p.write_text(s)
print(f"{dst} ready")
