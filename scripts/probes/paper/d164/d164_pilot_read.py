"""d164_pilot_read.py LOG... -- D164 step 0's reading (CPU): the [probe] lines of replay_probe.py runs.

Per window with an accepted candidate: three evaluations of the committed control, a (the line search's), x (the
probe's fresh one, the commit's protocol) and c (the commit), the window's taped pairs and the start pair. All
differences relative to max(|E_a|, floor) (floor = 1 / the run's unit ratio, from the log), as the check reads them.
  P0a: c - a above 10 x D164's measure (max(start pair, taped pairs, ulp(E_a))) and above the tag's tolerance
       max(1e-7, 10 x start pair);
  P0b: |c - x| above 10 x D164's measure;
  P0c: where c - a lies by term (the geometry energy lv, the released motion lstab, the cleanup, lambda x render as the
       rest of E), and the medians of |c - a|, |c - x|, the taped pairs and the start pair.
Stop rules (pre-registered): median |c - a| > 10 x median |c - x| on a mesh; |c - x| > 10 x the measure in > 1 %."""
import json
import re
import sys

import numpy as np


def read(path):
    txt = open(path).read()
    m = re.search(r"D_vol legacy\(\d+\^3\)/density = ([0-9.eE+-]+)", txt)
    floor = 1.0 / float(m.group(1))
    rows = [json.loads(s[len("[probe] "):]) for s in txt.splitlines() if s.startswith("[probe] ")]
    return floor, rows


def main():
    for path in sys.argv[1:]:
        floor, rows = read(path)
        W = []
        for r in rows:
            a, x, c = r.get("a"), r.get("x"), r.get("c")
            if not (a and x and c):
                continue
            Ea = a["E_ls"]
            sc = max(abs(Ea), floor)
            ulp = float(np.spacing(np.float32(abs(Ea)))) / sc
            taped = r["taped"]
            meas = max([r["replay_rel"] or 0.0, ulp] + taped)
            tag_tol = max(1e-7, 10.0 * (r["replay_rel"] or 0.0))
            d = {k: (c[k] - a[k]) / sc for k in ("lv", "lstab", "cleanup")}
            d["render"] = (c["E"] - a["E"]) / sc - sum(d.values())
            W.append(dict(ca=(c["E"] - Ea) / sc, cx=(c["E"] - x["E"]) / sc, xa=(x["E"] - Ea) / sc, meas=meas, ulp=ulp,
                          start=r["replay_rel"] or 0.0, taped=max(taped) if taped else None, n_taped=len(taped),
                          tag_null=(c["E"] - Ea) / sc > tag_tol, new_null=(c["E"] - Ea) / sc > 10 * meas,
                          budget=bool(r["iters_done"]) and max(r["iters_done"]) == 7, d=d, null=r.get("null")))
        n = len(W)
        med = lambda v: float(np.median(np.abs(v))) if len(v) else float("nan")  # noqa: E731
        ca, cx = [w["ca"] for w in W], [w["cx"] for w in W]
        print(f"== {path}: {len(rows)} probe lines, {n} windows with an accepted candidate, floor {floor:.3g}")
        print(f"  P0a: tag tolerance exceeded {sum(w['tag_null'] for w in W)} / {n} "
              f"({100 * sum(w['tag_null'] for w in W) / max(n, 1):.2f} %); D164 exceeded {sum(w['new_null'] for w in W)} / {n} "
              f"({100 * sum(w['new_null'] for w in W) / max(n, 1):.2f} %)")
        bx = sum(abs(w["cx"]) > 10 * w["meas"] for w in W)
        print(f"  P0b: |c - x| > 10 x measure {bx} / {n} ({100 * bx / max(n, 1):.2f} %)")
        print(f"  medians: |c - a| {med(ca):.3g}, |c - x| {med(cx):.3g}, |x - a| {med([w['xa'] for w in W]):.3g}, "
              f"start {med([w['start'] for w in W]):.3g}, taped max {med([w['taped'] for w in W if w['taped'] is not None]):.3g}, "
              f"ulp {med([w['ulp'] for w in W]):.3g}, D164 measure {med([w['meas'] for w in W]):.3g}")
        print(f"  stop rule 1 (median |c-a| > 10 median |c-x|): {'STOP' if med(ca) > 10 * med(cx) else 'no'}; "
              f"stop rule 2 (P0b > 1 %): {'AMEND' if bx > 0.01 * n else 'no'}")
        print(f"  c - a: mean {np.mean(ca):.3g}, positive in {sum(v > 0 for v in ca)} / {n}; c - x positive in "
              f"{sum(v > 0 for v in cx)} / {n}, exactly 0 in {sum(v == 0 for v in cx)}; c - a exactly 0 in {sum(v == 0 for v in ca)}")
        for k in ("lv", "lstab", "cleanup", "render"):
            v = [w["d"][k] for w in W]
            print(f"  P0c term {k:8s}: median |c - a| {med(v):.3g}, share of sum |.| {np.sum(np.abs(v)) / max(sum(np.sum(np.abs([w['d'][q] for w in W])) for q in ('lv', 'lstab', 'cleanup', 'render')), 1e-300):.2f}")
        q = lambda v: " ".join(f"{t:.3g}" for t in np.quantile(v, [.5, .9, .99, 1.0])) if len(v) else "-"  # noqa: E731
        print(f"  ratios (c - a) / measure q50 q90 q99 max: {q([w['ca'] / w['meas'] for w in W])}; "
              f"|c - x| / measure: {q([abs(w['cx']) / w['meas'] for w in W])}")
        print(f"  windows ended by the budget (no taped pair at the committed control): {sum(w['budget'] for w in W)} / {n}; "
              f"taped pairs a window: median {np.median([w['n_taped'] for w in W]) if W else 0}")
        big = sorted(W, key=lambda w: -w["ca"] / w["meas"])[:5]
        for w in big:
            print(f"    worst: c-a {w['ca']:.3g} c-x {w['cx']:.3g} meas {w['meas']:.3g} start {w['start']:.3g} taped {w['taped']} "
                  f"terms {json.dumps({k: float(f'{v:.3g}') for k, v in w['d'].items()})} null {w['null']}")


if __name__ == "__main__":
    main()
