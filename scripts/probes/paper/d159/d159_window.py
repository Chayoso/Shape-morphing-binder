"""d159_window.py OUT_JSONL LABEL=RUN_JSON ... -- D159 (the coordinator relaying D160's finding, 2026-10-22): is the render
starved after arrival? Per window of a run's history: lambda, the render's gradient on the controls weighted by lambda
(lambda |g_render|) against the physics' (|g_phys|), the render's share g_share, the control step taken (step_norm),
the body's move (the mean particle displacement over the window, pitches), and the decisions (accepted / rejected
iterations, null commit, outer rejection, the commit reason, the stop). Arrival: the first window by which 90 % of the
run's whole move has been made (from the data, the same rule for every run). Summaries before and after arrival
(medians; counts) and the last ten windows. The pitch from the run's target sample (its JSON's tgt_nn_spacing: a = 0.708
x the median 8th-neighbour distance is not in the JSON, so the move is also given in the target's nearest-neighbour
spacing)."""
import json
import sys

import numpy as np


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    for spec in specs:
        label, path = spec.split("=", 1)
        js = json.load(open(path))
        arm = next(iter(js["arms"].values()))
        hist = [h for h in arm["history"] if "animation" in h]
        sp = float(arm["metrics"].get("tgt_nn_spacing") or 1.0)
        mv = np.array([h.get("move") or 0.0 for h in hist])
        cum = np.cumsum(mv)
        k_arr = int(np.argmax(cum >= 0.9 * cum[-1])) if cum[-1] > 0 else 0
        rows = []
        for i, h in enumerate(hist):
            lam, gr, gp = h.get("lambda"), h.get("g_rend_norm"), h.get("g_phys_norm")
            rows.append(dict(w=i, after=i > k_arr, lam=lam, lam_gr=(lam * gr if lam is not None and gr is not None else None), gp=gp,
                             share=h.get("g_share"), step=(h.get("alpha_last") * h.get("grad_norm") if h.get("alpha_last") is not None and h.get("grad_norm") is not None else None), alpha=h.get("alpha_last"), move_sp=(h.get("move") or 0.0) / sp,
                             acc=h.get("accepted"), rej=h.get("rejected"), null=int(bool(h.get("null_commit"))),
                             outer_rej=int(bool(h.get("outer_rejected"))), reason=h.get("commit_reason"),
                             stop=h.get("actual_stop"), kin=h.get("kin")))

        def summ(sel):
            r = [x for x in rows if sel(x)]
            if not r:
                return None
            return dict(windows=len(r), lam=med([x["lam"] for x in r]), lam_gr=med([x["lam_gr"] for x in r]), gp=med([x["gp"] for x in r]),
                        share=med([x["share"] for x in r]), step=med([x["step"] for x in r]), alpha=med([x["alpha"] for x in r]), move_sp=med([x["move_sp"] for x in r]),
                        move_sp_sum=float(sum(x["move_sp"] for x in r)), null=sum(x["null"] for x in r),
                        outer_rej=sum(x["outer_rej"] for x in r), rejected_iters=sum(x["rej"] or 0 for x in r),
                        accepted_iters=sum(x["acc"] or 0 for x in r))
        row = dict(label=label, json=path, windows=len(rows), arrival_window=k_arr, spacing=sp,
                   before=summ(lambda x: not x["after"]), after=summ(lambda x: x["after"]), last10=summ(lambda x: x["w"] >= len(rows) - 10),
                   end_reason=[x["reason"] for x in rows[-3:]], stop=[x["stop"] for x in rows[-3:]], per_window=rows)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        b, a = row["before"], row["after"]
        g = lambda v, s=".3g": "-" if v is None else format(v, s)   # noqa: E731
        fmt = lambda d: (f"lam {g(d['lam'])}, lam|g_r| {g(d['lam_gr'])} vs |g_p| {g(d['gp'])}, share {g(d['share'], '.2f')}, step alpha|g| {g(d['step'])} (alpha {g(d['alpha'])}), "
                         f"move/window {d['move_sp']:.4f} sp (sum {d['move_sp_sum']:.2f}), null {d['null']}, outer rejected {d['outer_rej']}, "
                         f"iterations accepted/rejected {d['accepted_iters']}/{d['rejected_iters']}") if d else "-"    # noqa: E731
        print(f"{label}: {len(rows)} windows, arrival (90 % of the move) at window {k_arr}")
        print(f"   before: {fmt(b)}")
        print(f"   after:  {fmt(a)}")
        print(f"   end: reasons {row['end_reason']} stop {row['stop']}")


if __name__ == "__main__":
    main()
