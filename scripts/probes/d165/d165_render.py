"""d165_render.py OUT_DIR PATTERN_NPZ LABEL:PREFIX [LABEL:PREFIX ...] -- D165's close-ups (CPU, Embree via d157_lib,
D163's look and camera: d163_render.render_rake, the key light grazing, the close-up view of d163_render.closeup):
per run and per half of the front word, two tiles drawn by the same camera and light:
  plaque  the plaque's face after the settle (stamp.py's parents_settle), turned back to the run's frame (the word on
          +z, reading as in D163's close-ups);
  print   the slab's top layer after the lift (the particles within 0.35 wu of the top and 0.3 wu of the word's box),
          reflected across the plane y = z of the stamp's frame ((x, y, z) -> (x, z, y): the print, a mirror image of
          the stamp, then reads; a heightfield stays a heightfield) so its top faces +z like the plaque's face.
Both are display surfaces of their particles (the zbs field at the particles' own spacing: the plaque's a0, the slab's
s). OUT_DIR/<tag>_<plaque|print>_<left|right>.png (1920 x 1080) and OUT_DIR/closeup_sheet.jpg (a row per run:
plaque left, plaque right, print left, print right)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d163")
import d157_lib as D7  # noqa: E402
from d163_render import label, render_rake  # noqa: E402
from PIL import Image  # noqa: E402

out, pat_path = Path(sys.argv[1]), sys.argv[2]
items = [s.split(":", 1) for s in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
pat = np.load(pat_path, allow_pickle=True)
c, tex, texel, u0, v0 = pat["c_c"], pat["tex"], float(pat["texel"]), float(pat["u0"]), float(pat["v0"])
X0, X1 = c[0] + u0, c[0] + u0 + tex.shape[0] * texel
Z0, Z1 = c[1] + v0, c[1] + v0 + tex.shape[1] * texel
half = 0.5 * (X1 - X0)
W, H = 1920, 1080
rows = []
for lab, prefix in items:
    tag = lab.split(" ")[0]
    z = np.load(prefix + "_end.npz", allow_pickle=True)
    info = json.loads(str(z["info"]))
    R = np.asarray(z["R"], np.float64)
    P = np.asarray(z["parents_settle"], np.float64) @ R                    # stamp -> run: x R (R orthogonal)
    face = P[:, 2].max()
    S = np.asarray(z["slab_end"], np.float64)
    keep = ((S[:, 1] > -0.35) & (S[:, 0] > X0 - 0.3) & (S[:, 0] < X1 + 0.3) & (S[:, 2] > Z0 - 0.3) & (S[:, 2] < Z1 + 0.3))
    S = S[keep][:, [0, 2, 1]]                                               # the print reflected to read, its top on +z
    surfs = [("plaque", D7.field_mesh("zbs", P, float(info["a0"]))[0], face),
             ("print", D7.field_mesh("zbs", S, float(info["s"]))[0], S[:, 2].max())]
    tiles = []
    for kind, surf, zf in surfs:
        for hname, sx in (("left", -0.5), ("right", 0.5)):
            look = np.array([0.5 * (X0 + X1) + sx * half, 0.5 * (Z0 + Z1), zf])
            dist = 1.6 * half * 1.4
            img = render_rake(surf, look, dist, (0.30, 0.20), W, H, 0.55 * half / dist, rake=74.0)
            Image.fromarray(img).save(out / f"{tag}_{kind}_{hname}.png")
            tiles.append(label(img, lab if hname == "left" else "",
                               (("the plaque's face (settled, controls off)" if kind == "plaque"
                                 else "the slab's print after the lift (mirrored to read)") if hname == "left" else None),
                               scale=22))
    rows.append(np.concatenate(tiles, 1))
    print(json.dumps({"label": lab, "slab_particles": int(keep.sum()), "plaque_particles": len(P)}), flush=True)
sheet = np.concatenate(rows, 0)
Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / "closeup_sheet.jpg", quality=92)
print("wrote", out / "closeup_sheet.jpg")
