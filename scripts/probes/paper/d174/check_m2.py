"""check_m2.py -- D174 m2: Lattice.near_nodes chunked (1<<10, 1<<16 cells) against one chunk holding every cell, bit for bit,
on the 300k / 500k bodies of the probes' scene (the cached target samples, jittered) at the render's lattice pitch."""
import glob
import numpy as np
import torch
import physmorph  # noqa: F401
from physmorph.render.exterior import Lattice

ok = True
files = sorted(glob.glob("/data/relcfd/chayo/physmorph_v2/cache/samples/sample_d170smooth_dragon_300000_*.npz"))[:1]
files += sorted(glob.glob("/data/relcfd/chayo/physmorph_v2/cache/samples/sample_bunny_300000_*.npz"))[:1]
assert files, "no cached 300k samples"
for f in files:
    x = torch.as_tensor(np.load(f)["x"], device="cuda").float()
    x = (x - x.mean(0)) / (x.max(0).values - x.min(0).values).norm() * 8.0
    c = x.mean(0)
    lat = Lattice(c, 2.8 * float((x - c).norm(dim=1).max()))
    for pitch in (0.02, 0.025, 0.04):
        ref = lat.near_nodes(x, pitch, 2, chunk=1 << 40)
        for ch in (1 << 10, 1 << 16):
            got = lat.near_nodes(x, pitch, 2, chunk=ch)
            same = got.shape == ref.shape and torch.equal(got, ref)
            ok &= same
            print(f"{f.split('/')[-1][:40]} N={len(x)} pitch {pitch} chunk {ch}: nodes {len(ref)}, same {same}", flush=True)
print("NEAR_NODES BIT-FOR-BIT" if ok else "NEAR_NODES DIFFER")
