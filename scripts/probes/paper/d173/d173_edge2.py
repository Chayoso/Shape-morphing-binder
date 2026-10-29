"""d173_edge2.py OUT_JSONL JOBS -- D173: d173_edge.py's colour-edge widths for any two-colour scene whose paint has flat edges:
band (the plane y = tgt_y) and stripes (the planes s = tgt_s[0] and s = tgt_s[1], s = x . axis, red between them). Each
plane is read on the points nearer to it than to the other plane, its signed distance oriented so that red rises with it;
the row carries each plane's widths and their mean. The readings per plane are d173_edge.py's own (layer, logistic_width,
width, the discs and their kernel / nearest colours); JOBS lines as d173_edge.py's (`NAME FRAMES12`, `NAME@tgt`)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d173")
import d173_edge as E  # noqa: E402  (parses OUT_JSONL JOBS from sys.argv, as it always did)
from physmorph import gpu  # noqa: E402
from physmorph.render.exterior import Lattice, found_disc_albedo, make_field  # noqa: E402


def planes(app):
    if app["scene"] == "stripes":
        ax = np.asarray(app["axis"], np.float64)
        q1, q2 = app["tgt_s"]
        mid = 0.5 * (q1 + q2)
        return [("left", lambda X: (X @ ax - q1), lambda X: X @ ax < mid),
                ("right", lambda X: -(X @ ax - q2), lambda X: X @ ax >= mid)]
    ty = float(app["tgt_y"])
    return [("band", lambda X: X[:, 1] - ty, lambda X: np.ones(len(X), bool))]


def read(name, X, red, tgt, app):
    a = .708 * float(np.median(cKDTree(tgt[:20000]).query(tgt[:20000], 9)[0][:, 8])) * (20000 / len(tgt)) ** (1 / 3)
    row = dict(name=name, n=len(X), pitch=a, scene=app["scene"], planes={})
    lay = E.layer(X)
    Xt = gpu.tensor(X.astype(np.float32))
    alb = gpu.tensor(np.where(red[:, None], E.RED, 1.0).astype(np.float32))
    c = Xt.mean(0)
    lat = Lattice(c, 2.8 * float((Xt - c).norm(dim=1).max()))
    with torch.no_grad():
        f3 = make_field("zb", Xt, a, radius=3.0, offset=0.8)
        p, g, _, _ = lat.discs(f3, 0.46 * a, refine=False)
        k3 = found_disc_albedo(f3, p, alb, alb.mean(0), a).cpu().numpy()
    P = p.cpu().numpy().astype(np.float64)
    nn = cKDTree(X).query(P, 1, workers=16)[1]
    share = lambda col: np.clip((1.0 - col[:, 1]) / (1.0 - E.RED[1]), 0, 1)       # noqa: E731
    for nm, sd, near in planes(app):
        q = dict()
        XL, rl = X[lay], red[lay]
        m = near(XL)
        s = sd(XL[m]) / a
        q["particles"] = dict(zip(("width", "mid"), E.width(s, rl[m].astype(np.float64))[:2]))
        q["particles"].update(zip(("logistic", "logistic_mid"), E.logistic_width(s, rl[m].astype(np.float64))))
        q["scatter_2a"] = float((((s > 2) & ~rl[m]) | ((s < -2) & rl[m])).sum() / max(1, ((s > 2) | (s < -2)).sum()))
        mp = near(P)
        sp = sd(P[mp]) / a
        for k, v in (("discs_k3", share(k3)[mp]), ("discs_nn", red[nn][mp].astype(np.float64))):
            w_, m_, _ = E.width(sp, v)
            q[k] = dict(width=w_, mid=m_)
            q[k].update(zip(("logistic", "logistic_mid"), E.logistic_width(sp, v)))
        row["planes"][nm] = q
    for k in ("particles", "discs_k3", "discs_nn"):
        v = [q[k].get("logistic") for q in row["planes"].values()]
        row[k] = dict(logistic=float(np.mean(v)) if all(x is not None for x in v) else None,
                      width=None)
    v = [q["scatter_2a"] for q in row["planes"].values()]
    row["scatter_2a"] = float(np.mean(v))
    row["discs"] = int(len(P))
    return row


def main():
    out = open(E.out_path, "a")
    for job in E.jobs:
        name, path = job[0], job[1]
        z = np.load(path, allow_pickle=True)
        app = json.loads(str(z["appearance"]))
        tgt = np.asarray(z["tgt"], np.float64)
        if name.endswith("@tgt"):
            X, red = tgt, E.is_red(z["tgt_albedo"])
        else:
            X, red = np.asarray(z["frames"][-1], np.float64), E.is_red(z["src_albedo"])
        row = read(name, X, red, tgt, app)
        print(json.dumps({k: v for k, v in row.items() if k != "planes"}), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
