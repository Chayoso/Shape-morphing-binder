"""ls_read.py OUT_MD RUNDIR NAME [NAME ...] -- D175: the line-search trials of the keyframe segment windows split by term
(the keyframe-only ls_split record, solve.line_search): per trial the merit change d = d_lv (the end drift, the only
geometry inside a segment) + d_stab (the released motion) + lambda d_render, against the required decrease, with the
end speed and the released phase's mean speed before / after (in pitches per horizon T dt). And the replay check of every
segment commit (the segment's noise measure, pairs, verdict). CPU."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    out, rd, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    L = ["# D175: the segment windows' line-search trials by term (keyframe_compare column)", "",
         "Per trial: d = d_lv + d_stab + lambda d_count (the merit change; cleanup 0 in a segment), req = the required "
         "decrease. Speeds in pitches per horizon (T dt): the end speed |v_T| and the released phase's mean |v|, from -> to.",
         ""]
    agg = {"fail": [], "ok": []}
    for n in names:
        a = next(iter(json.load(open(f"{rd}/{n}.json"))["arms"].values()))
        z = np.load(f"{rd}/{n}_frames12.npz")
        tgt = np.asarray(z["tgt"], np.float64)
        pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
        hz = 20.0 / 240.0 / pitch
        L += [f"## {n}", "", "| window | outcome | trial (it) | ok | d / req | d_lv (end drift) | d_stab (released) | lambda d_count | "
              "end speed | released speed | blocker |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in a["history"]:
            if r.get("keyframe") != 0 or not r.get("ls_split"):
                continue
            out_ = r.get("null_reason") or ("hold" if r.get("hold") else "accepted")
            for t in r["ls_split"]:
                soft = t["d_lv"] + t["d_stab"]
                blk = ("-" if t["ok"] else "state check" if t["merit_ok"] else ("render rises" if t["d_lr_w"] > 0 else
                                             ("soundness: stab" if t["d_stab"] >= t["d_lv"] else "soundness: end drift")
                                             if soft > -t["d_lr_w"] else "short of req"))
                (agg["ok"] if t["ok"] else agg["fail"]).append(dict(t, blk=blk, hz=hz))
                L.append(f"| {r['animation'] + 1} | {out_} | {t['it']} | {t['ok']} | {t['d']:+.2e} / {t['req']:.1e} | "
                         f"{t['d_lv']:+.2e} | {t['d_stab']:+.2e} | {t['d_lr_w']:+.2e} | {t['v_end0'] * hz:.3f} -> "
                         f"{t['v_end'] * hz:.3f} | {t['v_rel0'] * hz:.3f} -> {t['v_rel'] * hz:.3f} | {blk} |")
        L.append("")
        for r in a["history"]:
            s = r.get("replay_split")
            if r.get("keyframe") == 0 and s:
                L.append(f"- window {r['animation'] + 1} replay: mismatch/E {(s['E_final'] - s['E_accept']) / abs(s['E_accept']):.2e}, "
                         f"tol/E {s['tol'] / abs(s['E_accept']):.2e}, noise {s.get('noise', float('nan')):.2e}, pairs {s.get('pairs')}, "
                         f"{'NULL' if r.get('null_reason') == 'commit_replay' else 'kept'}")
        L.append("")
    f, o = agg["fail"], agg["ok"]
    if f:
        kinds = {k: sum(1 for t in f if t["blk"] == k) for k in sorted({t["blk"] for t in f})}
        med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
        L += ["## Summary", "",
              f"Failed trials {len(f)}, by blocker: {kinds}.",
              f"Failed trials: median d_stab {med([t['d_stab'] for t in f]):+.2e}, d_lv {med([t['d_lv'] for t in f]):+.2e}, "
              f"lambda d_count {med([t['d_lr_w'] for t in f]):+.2e}; end speed x{med([t['v_end'] / max(t['v_end0'], 1e-12) for t in f]):.2f}, "
              f"released speed x{med([t['v_rel'] / max(t['v_rel0'], 1e-12) for t in f]):.2f} of the start's; released speed after "
              f"{med([t['v_rel'] * t['hz'] for t in f]):.3f} pitches / horizon.",
              f"Accepted trials {len(o)}: median d_stab {med([t['d_stab'] for t in o]):+.2e}, lambda d_count "
              f"{med([t['d_lr_w'] for t in o]):+.2e}; released speed x{med([t['v_rel'] / max(t['v_rel0'], 1e-12) for t in o]):.2f}, "
              f"after {med([t['v_rel'] * t['hz'] for t in o]):.3f} pitches / horizon."]
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
