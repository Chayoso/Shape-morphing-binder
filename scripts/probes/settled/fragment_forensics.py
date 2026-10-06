"""fragment_forensics.py DUMP_DIR [FIRST LAST] — which particles leave the body early in the morph, and what moved them,
from a run's per-window gradient dumps (`--grad_dump`, slimmed by tmp/d107_slim.py): windows FIRST..LAST (default 0..20).

At each window's committed end (xT_final): the particles' connected sets under a radius of 1.5 of the window's spacing
(the 16 nearest; label propagation on the GPU); every particle outside the largest set is a fragment (what the display
shows as Gaussians apart from the body). Per window: the fragments' count, how many were fragments at the window's
start (x0) already, how many were in the start's outer layer; for the new fragments against the body's outer layer: the
mean magnitude of the render push and the physics push (the weighted position gradients at xT0, along the outward
normal), and of what each channel alone moves them by (the linear-response end states on the stress control dFc and on
the layer control u, against their bases), and the window's whole move; the fragments' mean distance from the body's
largest set (spacings)."""
import glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
files = sorted(glob.glob(os.path.join(sys.argv[1], "win_*.npz")))
first, last = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 20)


def sets(x, sp):
    """(label (N,), size of each label's set (N,)): connected sets under the radius 1.5 sp among the 16 nearest."""
    d, nb = knn_self_torch(x, 17)
    d, nb = d[:, 1:], nb[:, 1:]
    nb = torch.where(d < 1.5 * sp, nb, torch.arange(len(x), device=dev)[:, None])
    lab = torch.arange(len(x), device=dev)
    for _ in range(400):
        new = torch.minimum(lab, lab[nb].min(1).values)
        new = new[new]                                          # pointer jumping
        if torch.equal(new, lab):
            break
        lab = new
    size = torch.bincount(lab, minlength=len(x))[lab]
    return lab, size


print("window | fragments (new / already at the start) | of the new: in the start's layer | mean |push| new vs body layer: "
      "render, physics | mean |move| new vs body layer: render dFc, render u, physics dFc, physics u, whole window | new: distance (sp)")
for path in files[first:last + 1]:
    z = np.load(path)
    t = lambda k: torch.as_tensor(z[k], device=dev).float()    # noqa: E731
    x0, xe, xT = t("x0"), t("xT_final"), t("xT0")
    sp = layer_spacing(x0)
    _, s0 = sets(x0, sp)
    lab_e, s_e = sets(xe, sp)
    big = s_e == s_e.max()
    frag_e, frag_0 = ~big, s0 < s0.max()
    new = frag_e & ~frag_0
    lay0, _ = layer_by_asymmetry(x0, sp)
    layT, nrmT = layer_by_asymmetry(xT, layer_spacing(xT))
    body = layT & big
    lam = float(z["lam_r"])
    push_r = ((-lam * (t("gx_sil") + t("gx_pbr"))) * nrmT).sum(1).abs()
    push_p = (-t("gx_phys") * nrmT).sum(1).abs()
    mv = {k: (t(a) - t(b)).norm(dim=1) / sp for k, (a, b) in dict(rd=("xT_rend", "xT_base"), ru=("xT_rend_u", "xT_base_u0"),
                                                                     pd=("xT_phys", "xT_base"), pu=("xT_phys_u", "xT_base_u0"),
                                                                     whole=("xT_final", "x0")).items()}
    ratio = lambda v: f"{float(v[new].mean()) / max(float(v[body].mean()), 1e-30):6.2f}" if int(new.sum()) else "     -"   # noqa: E731
    if int(new.sum()):
        dist = float(torch.cdist(xe[new][::max(1, int(new.sum()) // 2000)], xe[big][::20]).min(1).values.mean()) / sp
    else:
        dist = float("nan")
    print(f"{os.path.basename(path)[4:8]} | {int(frag_e.sum()):6d} ({int(new.sum()):6d} / {int((frag_e & frag_0).sum()):5d}) | "
          f"{int((new & lay0).sum()):6d} | {ratio(push_r)} {ratio(push_p)} | {ratio(mv['rd'])} {ratio(mv['ru'])} {ratio(mv['pd'])} "
          f"{ratio(mv['pu'])} {ratio(mv['whole'])} | {dist:5.2f}  lambda {lam:.3f} g_share {float(z['g_share']):.2f}", flush=True)
