"""d165_render.py OUT_DIR PATTERN_NPZ LABEL:PREFIX [LABEL:PREFIX ...] -- D165's close-ups of the slab's imprint (CPU,
Embree via d157_lib, D163's look: d163_render.render_rake, the key light grazing): per run the slab's top layer at the
end (stamp.py's PREFIX_end.npz, the particles within 0.35 wu of the top and 0.3 wu of the word's box) as its display
surface (the zbs field at the slab's spacing), seen from above (azimuth pi, elevation 1.15 rad: the image's right is -x,
its up +z), then flipped left-right so the word reads (an imprint is the stamp's mirror image); OUT_DIR/imprint_<label>.png
(3840 x 1080) and OUT_DIR/imprint_sheet.png (the runs stacked, labelled)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d163")
import d157_lib as D7  # noqa: E402
from d163_render import label, render_rake  # noqa: E402

out, pat_path = Path(sys.argv[1]), sys.argv[2]
items = [s.split(":", 1) for s in sys.argv[3:]]
out.mkdir(parents=True, exist_ok=True)
pat = np.load(pat_path, allow_pickle=True)
c, tex, texel, u0, v0 = pat["c_c"], pat["tex"], float(pat["texel"]), float(pat["u0"]), float(pat["v0"])
X0, X1 = c[0] + u0, c[0] + u0 + tex.shape[0] * texel
Z0, Z1 = c[1] + v0, c[1] + v0 + tex.shape[1] * texel
W, H = 3840, 1080
tiles = []
for lab, prefix in items:
    z = np.load(prefix + "_end.npz", allow_pickle=True)
    info = json.loads(str(z["info"]))
    s = float(info["s"])
    S = np.asarray(z["slab_end"], np.float64)
    keep = ((S[:, 1] > -0.35) & (S[:, 0] > X0 - 0.3) & (S[:, 0] < X1 + 0.3) & (S[:, 2] > Z0 - 0.3) & (S[:, 2] < Z1 + 0.3))
    surf, _ = D7.field_mesh("zbs", S[keep], s)
    look = np.array([0.5 * (X0 + X1), -0.05, 0.5 * (Z0 + Z1)])
    hw = 0.5 * (X1 - X0)
    dist = 2.2 * hw
    img = render_rake(surf, look, dist, (np.pi, 1.15), W, H, 1.08 * hw / dist, rake=74.0)
    img = np.ascontiguousarray(img[:, ::-1])
    from PIL import Image
    Image.fromarray(img).save(out / f"imprint_{lab.split(' ')[0]}.png")
    tiles.append(label(img, lab, "the slab after the plaque is lifted, seen from above (mirrored to read)", scale=14))
    print(json.dumps({"label": lab, "particles": int(keep.sum()), "s": s}), flush=True)
from PIL import Image  # noqa: E402
Image.fromarray(np.concatenate(tiles, 0)).save(out / "imprint_sheet.png")
print("wrote", out / "imprint_sheet.png")
