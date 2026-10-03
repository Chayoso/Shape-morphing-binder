"""near_band_probe.py ARCHIVE_NPZ RUN_JSON TERMS_DIR — D32: what holds the detached material that is left near the surface.

The end state's rendered detached particles beyond the sampling berth (single linkage at one layer spacing, the body =
the largest set), split by what the layer makes of each at the start of a window: off the layer (the asymmetry of
its neighbourhood is under the threshold: it has material on both sides), on the layer as a single particle (relaxed
against its neighbours), on the layer as a member of a group (not relaxed, u along the direction away from the
surroundings). Per class, over the last windows of the run (term dump with the channel record): how many; the
distance to the target then and at the end; the asymmetry in units of the threshold; the share inside u's transport
gate; the control u in layer spacings and its sign along the direction to the target; the displacement toward the
target per window by the MPM advection, by u and by the rest; the share on which the near band and the spray
cleanup have a gradient, and the size of their pull against the all-particle rms gradient."""
import glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_spacing, outside_neighbours  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames, n_del = z["frames"], int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
berth = float(cfg.get("nn_berth_k", 1.97)) * sp
tknn = gpu.KNN(tgt)
xe = torch.as_tensor(np.asarray(frames[n_del - 1], np.float32), device=dev)
d_e, nb_e = knn_self_torch(xe, 33)
group_e, _ = detached_groups(d_e, nb_e, layer_spacing(xe))
dist_e = tknn.query(xe, 1)[0][:, 0].float()
fl = torch.nonzero((group_e > 0) & (live_support(d_e, cov, sp) > 0) & (dist_e > berth)).squeeze(1)
files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
print(f"N {len(xe)}, end frame {n_del - 1}, {len(files)} dumped windows; target spacing sp {sp:.4f} wu, berth {berth / sp:.2f} sp; "
      f"rendered detached particles beyond the berth at the end: {len(fl)} (median distance {float(dist_e[fl].median()) / sp:.1f} sp, max {float(dist_e[fl].max()) / sp:.1f})")
T = ("ot", "surf", "near", "spray", "rend")
for k in [k for k in (len(files) - 20, len(files) - 12, len(files) - 6, len(files) - 2) if k >= 0]:
    f = np.load(files[k])
    x0 = torch.as_tensor(f["c_x0"], device=dev)
    lmask, ug, u = (torch.as_tensor(f["c_" + c], device=dev) for c in ("lmask", "ug", "u"))
    ch = {c: torch.as_tensor(f["c_" + c], device=dev) for c in ("d_adv", "d_u", "d_rest")}
    g = {t: torch.as_tensor(f["g_" + t], device=dev) for t in T}
    rms_all = float(sum(g.values()).pow(2).sum(1).mean().sqrt())
    spacing = layer_spacing(x0)
    d_a, nb_a = knn_self_torch(x0, 33)
    group, size = detached_groups(d_a, nb_a, spacing)
    off = x0 - x0[nb_a[:, 1:]].mean(1)
    member = (group > 0) & (size > 1) & (size <= 512)
    q = torch.nonzero(member).squeeze(1)
    if len(q):
        nb_o, _, ok = outside_neighbours(x0, torch.arange(len(x0), device=dev), q, group, size, 32)
        cen = (x0[nb_o] * ok[..., None]).sum(1) / ok.sum(1, keepdim=True).clamp_min(1)
        off[q[ok.any(1)]] = (x0[q] - cen)[ok.any(1)]
    asym = off.norm(dim=1) / (0.5 * spacing)
    dq, it = tknn.query(x0, 1)
    n_t = torch.nn.functional.normalize(tgt[it[:, 0]] - x0, dim=1)
    dist = dq[:, 0].float()
    on = lmask > 0.5
    classes = (("off the layer            ", ~on), ("on the layer, single     ", on & ~member), ("on the layer, in a group ", on & member))
    print(f"\n== window {k} (start state): layer spacing {spacing / sp:.2f} sp; all-particle rms gradient {rms_all:.2e}")
    print("   class | particles (of the end's floaters) | detached then % | distance sp then -> at the end | asymmetry / threshold (median) | inside u's gate % | |u| in layer spacings (median) | "
          "toward the target this window, sp: advection, u, rest | near band on %, pull | spray on %, pull")
    for name, m_all in classes:
        m = fl[m_all[fl]]
        if len(m) == 0:
            print(f"   {name} | 0")
            continue
        tw = [float((ch[c][m] * n_t[m]).sum(1).mean()) / sp for c in ("d_adv", "d_u", "d_rest")]
        pull = lambda t: (100 * float((g[t][m].norm(dim=1) > 0).float().mean()), float((-g[t][m] * n_t[m]).sum(1).mean()) / rms_all)  # noqa: E731
        print(f"   {name} | {len(m):4d} | {100 * float((group[m] > 0).float().mean()):3.0f} | {float(dist[m].median()) / sp:4.1f} -> {float(dist_e[m].median()) / sp:4.1f} | {float(asym[m].median()):.2f} | "
              f"{100 * float((ug[m] > 0.5).float().mean()):3.0f} | {float(u[m].abs().median()) / spacing:.3f} | {tw[0]:+.3f} {tw[1]:+.3f} {tw[2]:+.3f} | "
              f"{pull('near')[0]:3.0f} %, {pull('near')[1]:+.2f} | {pull('spray')[0]:3.0f} %, {pull('spray')[1]:+.2f}")
