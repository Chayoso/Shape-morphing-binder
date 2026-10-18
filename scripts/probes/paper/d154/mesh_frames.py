"""mesh_frames.py OUT_DIR MESH:SEED [...] -- D154 (1): each mesh-seed's target mesh in the run's own target frame.

Run from repo_ts (the tag's sampler, its cached samples). load_normalized puts a mesh's cloud in its frame as
world = (oriented mesh - offset) x scale (prepare.prepare: the source sample at SEED, then the target at SEED + 1
matched to the source's volume); the target sample it returns must equal the run's own (the end1 archive's `tgt`) to 1e-6 wu
(bit for bit, or within float32 rounding: some seeds differ by <= 4.8e-7 wu, a scale of 1e-7), else the mesh-seed is refused. Writes OUT_DIR/MESH_SEED.npz: V (world, float64, duplicate vertices merged), F,
offset, scale, orient; one JSON line per mesh-seed (faces, watertight, the check)."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path.cwd()))
from physmorph.sampling.mesh import load_mesh, load_normalized  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[2:]:
    mesh, seed = spec.split(":")
    seed = int(seed)
    path = f"assets/{mesh}.obj"
    fill_src, fill_tgt, frame = {}, {}, {}
    _, v_src = load_normalized("assets/isosphere.obj", 300000, seed, return_volume=True, sample="stratified",
                               fill=fill_src)
    tgt, _ = load_normalized(path, 300000, seed + 1, match_volume=v_src, sample="stratified", return_volume=True,
                             frame=frame, fill=fill_tgt)
    own = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")["tgt"], np.float32)
    maxdiff = float(np.abs(tgt - own).max())
    same = maxdiff <= 1e-6                     # bit for bit, or float32 rounding of the volume match (<= 4.8e-7 wu seen)
    m = load_mesh(path)
    o = orient_name(path)
    V = np.asarray(m.vertices, np.float64)
    if o != "id":
        V = V @ rotation(o).T
    V = (V - frame["offset"]) * frame["scale"]
    t = trimesh.Trimesh(V, np.asarray(m.faces), process=True)        # duplicate vertices merged (armadilo is a soup)
    row = dict(mesh=mesh, seed=seed, same_as_run_target=same, maxdiff_wu=maxdiff, bit_for_bit=bool(maxdiff == 0.0), faces=len(t.faces), verts=len(t.vertices),
               watertight=bool(t.is_watertight), winding_consistent=bool(t.is_winding_consistent), orient=o,
               offset=[float(v) for v in frame["offset"]], scale=float(frame["scale"]),
               volume=float(t.volume) if t.is_watertight else None)
    if same:
        np.savez(out / f"{mesh}_{seed}.npz", V=np.asarray(t.vertices, np.float64), F=np.asarray(t.faces, np.int32),
                 offset=np.asarray(frame["offset"]), scale=np.float64(frame["scale"]), orient=np.str_(o))
    print(json.dumps(row), flush=True)
