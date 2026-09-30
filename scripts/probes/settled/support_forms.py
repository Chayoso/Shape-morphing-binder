"""support_forms.py LABEL=ARCHIVE.npz[:E] ... — the log-form support penalty against the bounded ratio form, measured at
an end state (read-only; nothing in the pipeline changes).

Per particle i: r_i = s_i / f_i (kernel density over the floor; global floor and target-referenced floor), the log
form [-log r]_+^2 and the ratio form [1 - r]_+^2, both times the kernel radius squared as in TransportSupport.
Printed: the deficit distribution (share with r < 1, r quantiles among them), B under each form, the share of B held
by the top particle and by r < 0.1, the bulk ratio new/old among 0.3 <= r < 1 (penalty and dL/dr), the outer
coupling's support-gradient weight w (E / (E + w B))^2 under each form when E is given, and the five largest log-form
particles: r, distance to the nearest target point (target spacings), and the W1 isolation-gate weight.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import sys

import numpy as np
import torch

from physmorph import gpu
from physmorph.losses.support import TransportSupport
from physmorph.losses.volumetric import isolation_gate
from physmorph.pipeline import PipelineConfig

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from thin_regions import npz_member  # noqa: E402


def forms(sup, x):
    logr = sup.log_density(x).double() - torch.as_tensor(sup.floor(x), dtype=torch.float64, device=x.device)
    r = logr.exp()
    old = sup.radius ** 2 * torch.relu(-logr).square()
    new = sup.radius ** 2 * torch.relu(1.0 - r).square()
    return r, old, new


def analyse(label, path, E):
    cfg = PipelineConfig()
    z = np.load(path, allow_pickle=True)
    frames = npz_member(path, "frames")
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
    x = gpu.tensor(np.asarray(frames[dn - 1], np.float32))
    tgt = gpu.tensor(np.asarray(z["tgt"], np.float32))
    sp = gpu.median(gpu.knn(tgt, 2)[0][:, 1])
    print(f"\n== {label}: N {len(x)}")
    for ref in (False, True):
        sup = TransportSupport(tgt, cfg.support_weight, target_ref=ref)
        r, old, new = forms(sup, x)
        d = r < 1
        rq = torch.quantile(r[d], torch.tensor([.01, .1, .5, .9], dtype=r.dtype, device=r.device)) if bool(d.any()) else None
        Bo, Bn = float(old.mean()), float(new.mean())
        top = torch.topk(old, 5)
        bulk = (r >= .3) & (r < 1)
        ratio_pen = float(new[bulk].sum() / old[bulk].sum().clamp_min(1e-300)) if bool(bulk.any()) else float("nan")
        rb = r[bulk]
        ratio_grad = float(torch.median((2 * (1 - rb)) / (2 * (-rb.log()) / rb))) if bool(bulk.any()) else float("nan")
        print(f"  floor {'target-referenced' if ref else 'global median'}: {100 * float(d.double().mean()):.2f} % below the "
              f"floor; r quantiles (1/10/50/90 %) {[round(float(v), 3) for v in rq] if rq is not None else '-'}")
        print(f"    B log form {Bo:.3e}  ratio form {Bn:.3e}  | top particle holds {100 * float(top.values[0]) / (Bo * len(x)):.1f} % "
              f"of the log-form B, r < 0.1 holds {100 * float(old[r < .1].sum()) / (Bo * len(x)):.1f} %")
        print(f"    bulk 0.3 <= r < 1 ({int(bulk.sum())} particles): ratio/log penalty {ratio_pen:.2f}, median ratio of dL/dr {ratio_grad:.2f}")
        print(f"    per-particle max / p99 / median of the deficient: log {float(old.max()):.3e} / "
              f"{float(torch.quantile(old[d], .99)) if bool(d.any()) else 0:.3e} / {float(old[d].median()) if bool(d.any()) else 0:.3e}"
              f"   ratio {float(new.max()):.3e} / {float(torch.quantile(new[d], .99)) if bool(d.any()) else 0:.3e} / "
              f"{float(new[d].median()) if bool(d.any()) else 0:.3e}")
        if E is not None:
            w = cfg.support_weight
            wo, wn = w * (E / (E + w * Bo)) ** 2, w * (E / (E + w * Bn)) ** 2
            print(f"    outer coupling support-gradient weight w(E/(E+wB))^2: log {wo:.3f}  ratio {wn:.3f}  (E {E:.3e})")
        if not ref:
            dn_t = gpu.KNN(tgt).query(x[top.indices], 1)[0][:, 0] / sp
            gate = isolation_gate(x, cfg.dt_iso_lo, cfg.dt_iso_hi)[top.indices]
            print("    five largest (log form): " + "; ".join(
                f"pen {float(p):.1f} r {float(r[i]):.1e} to-target {float(t):.1f} sp W1-gate {float(g):.2f}"
                for p, i, t, g in zip(top.values, top.indices, dn_t, gate)))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        label, rest = a.split("=", 1)
        path, E = (rest.rsplit(":", 1) + [None])[:2] if rest.count(":") else (rest, None)
        analyse(label, path, float(E) if E else None)
