"""D145 B12 end check (B5's endcheck.py on the B12 archive): the t = 1 sample against the independent 300k sample (d90)
and the pipeline's own seed-97 target sample (physmorph.metrics.sil_iou, chamfer); t = 0 against the pipeline's source
sample; the floor (own against independent). Frames that could not be sampled (NaN) are reported as such.

  $PY endcheck_b12.py OUT_JSONL NAME TAG
"""
import json
import sys

from physmorph import gpu  # noqa: F401  (before torch)
import numpy as np
from physmorph import metrics

out, name, tag = sys.argv[1], sys.argv[2], sys.argv[3]
B = "/data/relcfd/chayo/physmorph_v2/baselines/buonomo"
z = np.load(f"/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12/{name}300k_{tag}_frames12.npz")
x, own = np.asarray(z["frames"][-1], np.float32), np.asarray(z["tgt"], np.float32)
x0 = np.asarray(z["frames"][0], np.float32)
src = np.asarray(np.load(f"{B}/inputs/{name}/pipeline_samples.npz")["src"], np.float32)
other = np.asarray(np.load(f"/data/relcfd/chayo/physmorph_v2/output/gpu/d90/{name}_ind300k.npz")["tgt"], np.float32)


def pair(a, b):
    if not np.isfinite(a).all():
        return dict(sil=None, chamfer=None, note="frame not sampled")
    return dict(sil=metrics.sil_iou(a, b), chamfer=metrics.chamfer(a, b))


row = dict(tag=f"{name}300k_{tag}", own=pair(x, own), ind=pair(x, other), floor=pair(own, other), start=pair(x0, src))
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)
