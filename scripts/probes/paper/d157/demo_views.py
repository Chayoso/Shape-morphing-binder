"""demo_views.py OUT_PNG -- D157 demo: a contact sheet of the exact mesh's silhouettes (bunny, dragon, seed 97, 300k
frame) from the 46 held-out directions at 256 px, numbered, to choose the views the overlays show."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import d157_read as R  # noqa: E402

out = sys.argv[1]
fig, axes = plt.subplots(2 * 6, 8, figsize=(16, 24))
for k, mesh in enumerate(("bunny", "dragon")):
    z = np.load(f"{R.G}/d145t_b1/{mesh}300k_L97_end1.npz")
    ext = float(np.abs(np.asarray(z["tgt"])).max()) * 1.25
    m = np.load(f"{R.MESHDIR['300k']}/{mesh}_97.npz")
    t, _ = R.close_holes(np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64))
    s = R.Surf(np.asarray(t.vertices), np.asarray(t.faces))
    for i, v in enumerate(R.HELD):
        ax = axes[6 * k + i // 8, i % 8]
        mask = s.occluded(R.ortho_rays(v, ext, 256)).reshape(256, 256)
        ax.imshow(mask.T[::-1], cmap="Greys", vmin=0, vmax=1.6)
        ax.set_title(f"{mesh} {i}: az {np.degrees(v[0]) % 360:.0f} el {np.degrees(v[1]):.0f}{' far' if R.FAR[i] else ''}",
                     fontsize=7)
        ax.axis("off")
    for j in range(len(R.HELD), 48):
        axes[6 * k + j // 8, j % 8].axis("off")
fig.tight_layout()
fig.savefig(out, dpi=70)
