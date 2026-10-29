"""d173_close3.py OUT_DIR BASE N RELIEF_OBJ LABEL:NPZ [LABEL:NPZ ...] -- D173: one raking close-up sheet (84 deg, D163's
renderer, the true camera) of the picture target (the relief mesh) and each run's end state, one camera for all: the look at
the centroid of the relief's pattern (P > 0.5, its .pattern.npz), the view along its mean normal turned a quarter, as
tmp/d170/d170_render.py close does. OUT_DIR/<label>.png and OUT_DIR/sheet_rake84.jpg (the tiles side by side)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d163")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
import d170_render as DR  # noqa: E402
from d161_pattern import Pattern  # noqa: E402

out, base, n, relief = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
items = [s.split(":", 1) for s in sys.argv[5:]]
out.mkdir(parents=True, exist_ok=True)
frame, x_t, c, rad, a = DR.setup(base, n, 97)
rs = DR.mesh_surf(relief, frame)
V = rs.V
import trimesh  # noqa: E402
vn = np.asarray(trimesh.Trimesh(V, rs.F, process=False).vertex_normals)
sub = np.random.default_rng(0).choice(len(V), min(400000, len(V)), replace=False)
P = Pattern.load(relief + ".pattern.npz").at(V[sub], vn[sub], zpts=V[sub])
hot = sub[P > 0.5]
look = V[hot].mean(0)
view = DR.view_of(vn[hot].mean(0))
view = (view[0] + 0.25, max(view[1], -1.2))
dist = 1.6 * rad
ext = float(np.linalg.norm(V[hot] - look, axis=1).max())
tan_h = 0.6 * ext / dist
W, H = 1280, 1080
from PIL import Image  # noqa: E402
tiles = [("picture target (the scale relief)", rs)]
for lab, npz in items:
    z = np.load(npz, allow_pickle=True)
    tiles.append((lab, DR.display(np.asarray(z["frames"][-1], np.float64), a)))
row = []
for lab, s in tiles:
    img = DR.render_rake(s, look, dist, view, W, H, tan_h, rake=84.0)
    Image.fromarray(img).save(out / f"{lab.split(' (')[0].replace(' ', '_').replace(':', '')}.png")
    row.append(DR.label(img, lab, "raking light 84 deg, end state", scale=22))
sheet = np.concatenate(row, 1)
Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / "sheet_rake84.jpg", quality=92)
print("done", out, flush=True)
