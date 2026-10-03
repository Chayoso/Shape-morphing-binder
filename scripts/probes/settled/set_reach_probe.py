"""set_reach_probe.py STATE [...] — D45: what tells a set at the surface from a set in the air.

STATE is ARCHIVE_NPZ:RAW (a frame of a run's archive or of a key-frame file) or TERMS_NPZ@ARCHIVE_NPZ (the window-start
state c_x0 of a term dump, the target read from the archive).

For the rendered particles of the sets that are not connected to the body within one layer spacing (layer.detached_groups):
by distance to the target (within the berth of 1.97 target spacings, to one loss cell of 4.4, beyond), three readings of
"measured against what":
  gap: the set's smallest distance to a particle of the body, in layer spacings (the relaxation's weight is a Gaussian
    of layer_h_sp = 2 layer spacings);
  own weight: for the set's layer particles, the share of their ordinary relaxation row (24 nearest same-side layer
    particles, the Gaussian weight; the row as it would be were no set detached) that lies on members of their own set;
  dense: 8th-neighbour distance under 1.2 coverage radii (the display draws it opaque, the spray gate is shut)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

dev = torch.device("cuda")
GAPS = (1.5, 2.0, 3.0, 4.0)
for spec in sys.argv[1:]:
    if "@" in spec:
        terms, arch = spec.split("@")
        zt = np.load(terms)
        x = torch.as_tensor(zt["c_x0"] if "c_x0" in zt.files else zt["x"], device=dev)   # dumps before D21: the window's end state
        tgt_np = np.load(arch, allow_pickle=True)["tgt"]
        name = "/".join(terms.split("/")[-3:])
    else:
        arch, raw = spec.rsplit(":", 1)
        z = np.load(arch, allow_pickle=True)
        raws = [int(v) for v in z["raws"]] if "raws" in z.files else None
        x = torch.as_tensor(np.asarray(z["frames"][raws.index(int(raw)) if raws else int(raw)], np.float32), device=dev)
        tgt_np, name = z["tgt"], f"{arch.split('/')[-2]} raw {raw}"
    tgt = torch.as_tensor(np.asarray(tgt_np, np.float32), device=dev)
    with torch.inference_mode():
        td = knn_self_torch(tgt, 9)[0]
        sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
        d, nb = knn_self_torch(x, 33)
        a = layer_spacing(x)
        group, size = detached_groups(d, nb, a)
        det = group > 0
        shown = live_support(d, cov, sp) > 0
        dist = gpu.KNN(tgt).query(x, 1)[0][:, 0].float() / sp
        ids = torch.nonzero(det).squeeze(1)
        to_body = gpu.KNN(x[~det]).query(x[ids], 1)[0][:, 0].float() / a
        G = int(group.max()) + 1
        gap_set = torch.full((G,), float("inf"), device=dev).scatter_reduce(0, group[ids], to_body, "amin")
        gap = torch.zeros(len(x), device=dev); gap[ids] = gap_set[group[ids]]
        mask, _, nbr, w = layer_relax_data(x, a, group_query=1)             # the ordinary rows: no set detached
        own = (w * (group[nbr] == group[:, None]).float()).sum(1)
        lay = mask > .5
        dense = d[:, 8] / cov < 1.2
    print(f"{name}: layer spacing {a / sp:.2f} target spacings; {int(det.sum())} detached particles in {G - 1} sets, {int((det & shown).sum())} rendered")
    print("   band | rendered | by the set's gap to the body, layer spacings: <= 1.5, 1.5-2, 2-3, 3-4, > 4 | on the layer | own weight of their ordinary row: median, share above one half | "
          "dense | dense with gap > 2 | in sets of 24 and more")
    for band, m in (("within the berth", dist <= 1.97), ("to one loss cell", (dist > 1.97) & (dist <= 4.4)), ("beyond a loss cell", dist > 4.4)):
        r = det & shown & m
        n = int(r.sum())
        if not n:
            print(f"   {band:18s} | 0"); continue
        bins, lo = [], 0.0
        for hi in GAPS + (float("inf"),):
            bins.append(int((r & (gap > lo) & (gap <= hi)).sum())); lo = hi
        rl = r & lay
        ow = own[rl]
        print(f"   {band:18s} | {n:5d} | " + ", ".join(f"{b:5d}" for b in bins) + f" | {int(rl.sum()):5d} | "
              + (f"{float(ow.median()):.2f}, {100 * float((ow > .5).float().mean()):3.0f} %" if len(ow) else "  -  ")
              + f" | {int((r & dense).sum()):4d} | {int((r & dense & (gap > 2)).sum()):4d} | {int((r & (size >= 24)).sum()):4d}")
