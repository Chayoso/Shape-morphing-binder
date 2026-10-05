"""momentum_probe.py LABEL=FRAMES_NPZ [...] — momentum over a whole run, from the kept frames' positions (uniform mass,
no external force, a start at rest: conserved linear momentum keeps the centre of mass where it is, conserved angular
momentum keeps the mean of r x dx at zero). One JSON row per pair of kept frames (state: the later frame's raw index;
com: the centre of mass's displacement from the first frame; lin, ang: the net over the gross linear move |mean dx| /
mean |dx| and angular move |mean r x dx| / mean |r| |dx| about the centre of mass; rot: the net rotation summed so far,
degrees; move: the mean particle move of the pair), lengths in display pitches a. Then per run one line: the centre of
mass's largest and final displacement and its path, the two ratios (largest, mean, mean of the last 20 pairs), the net
rotation over the run, and the mean particle move per pair over the last 20 pairs (what still moves at the end)."""
import json, sys
import numpy as np
import torch

dev = torch.device("cuda")
for spec in sys.argv[1:]:
    label, path = spec.split("=")
    z = np.load(path, allow_pickle=True)
    raws = [int(v) for v in z["raws"]]
    tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
    d = torch.cdist(tgt[::50], tgt).topk(9, largest=False).values[:, 8]
    a = .708 * float(d.median())
    fr = z["frames"]
    com0, prev, path_len, far, lin, ang, rot, still = None, None, 0., 0., [], [], torch.zeros(3, dtype=torch.float64, device=dev), []
    for i in range(len(fr)):
        x = torch.as_tensor(np.asarray(fr[i], np.float32), device=dev).double()
        c = x.mean(0)
        if prev is None:
            com0 = c
        else:
            dx = x - prev
            r = prev - prev.mean(0)
            gross = dx.norm(dim=1).mean()
            path_len += float((c - prev.mean(0)).norm())
            lin.append(float(dx.mean(0).norm() / gross.clamp_min(1e-30)))
            L = torch.cross(r, dx, dim=1).mean(0)
            ang.append(float(L.norm() / (r.norm(dim=1) * dx.norm(dim=1)).mean().clamp_min(1e-30)))
            rot += L / r.square().sum(1).mean() * 1.5          # the rigid rotation with that angular move (isotropic body)
            still.append(float(gross) / a)
            print(json.dumps(dict(run=label, state=raws[i], com=float((c - com0).norm()) / a, lin=lin[-1], ang=ang[-1],
                                  rot=float(torch.rad2deg(rot.norm())), move=still[-1])), flush=True)
        far = max(far, float((c - com0).norm()))
        prev = x
    end = float((prev.mean(0) - com0).norm())
    m = lambda v, k=None: float(np.mean(v[-k:] if k else v))
    print(f"{label}: {len(fr)} kept frames | centre of mass: largest displacement {far / a:.4f} a, at the end {end / a:.4f} a, path {path_len / a:.4f} a | "
          f"net / gross linear move: largest {max(lin):.2e}, mean {m(lin):.2e}, last 20 {m(lin, 20):.2e} | net / gross angular move: largest {max(ang):.2e}, mean {m(ang):.2e}, last 20 {m(ang, 20):.2e} | "
          f"net rotation over the run {float(torch.rad2deg(rot.norm())):.4f} deg | mean particle move per kept-frame pair, last 20: {m(still, 20):.4f} a", flush=True)
