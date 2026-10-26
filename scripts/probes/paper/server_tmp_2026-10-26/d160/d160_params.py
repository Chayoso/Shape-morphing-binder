"""d160_params.py -- the scenes' numbers on the tag's 40k bunny samples (frames12: frame 0 the source, tgt the target)."""
import json, sys
import numpy as np
from physmorph.pipeline.appearance import make_appearance, target_red, source_red
for path in sys.argv[1:]:
    z = np.load(path)
    src, tgt = np.asarray(z["frames"][0], np.float32), np.asarray(z["tgt"], np.float32)
    for sc in ("half", "band", "ears", "turn"):
        a = make_appearance(sc, src, tgt)
        print(json.dumps(dict(run=path.split("/")[-1], scene=sc, src_red=float(source_red(a.params, src).mean()),
                              tgt_red=float(target_red(a.params, tgt).mean()),
                              **{k: v for k, v in a.params.items() if k not in ("white", "red", "scene")})), flush=True)
