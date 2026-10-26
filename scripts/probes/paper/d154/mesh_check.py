"""mesh_check.py -- D154: the six meshes (watertight, faces) and each mesh-seed's frame recovered by load_normalized
(cached samples), checked against the run's own target sample in its end1 archive (bit for bit)."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path.cwd()))
from physmorph.sampling.mesh import load_normalized, load_mesh  # noqa: E402
from physmorph.sampling.orientation import orient_name  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
for spec in sys.argv[1:]:
    mesh, seed = spec.split(":")
    seed = int(seed)
    m = load_mesh(f"assets/{mesh}.obj")
    fill_src, fill_tgt, frame = {}, {}, {}
    src, v_src = load_normalized("assets/isosphere.obj", 300000, seed, return_volume=True, sample="stratified",
                                 fill=fill_src)
    tgt, v_tgt = load_normalized(f"assets/{mesh}.obj", 300000, seed + 1, match_volume=v_src, sample="stratified",
                                 return_volume=True, frame=frame, fill=fill_tgt)
    z = np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")
    own = np.asarray(z["tgt"], np.float32)
    row = dict(mesh=mesh, seed=seed, faces=len(m.faces), verts=len(m.vertices), watertight=bool(m.is_watertight),
               orient=orient_name(f"assets/{mesh}.obj"), offset=[float(v) for v in frame["offset"]],
               scale=float(frame["scale"]), same_as_end1=bool(np.array_equal(tgt, own)),
               maxdiff=float(np.abs(tgt - own).max()))
    print(json.dumps(row), flush=True)
