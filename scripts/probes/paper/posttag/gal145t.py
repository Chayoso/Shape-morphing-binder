"""gal145t.py OUT_MD [--dir d145t_b3] [--pre d145_b3] -- D145 B3 post-tag: the 40k gallery, render arm V against its
lambda = 0 twin W (seeds 97 / 99 / 101), as gal145.py read it before the tag, and the tag against the pre-tag gallery
(CPU only; gallery_ind.py's ind.log, the runs' JSON, d120_apart.py's and visible_apart.py's logs).

Per mesh: V's and W's independent-sample silhouette IoU (min / median / max), the floor (own target sample against the
independent one), V - W per seed, its three-seed mean, whether the seeds agree, the gain as a share of the twin's gap to
the floor; chamfer; the tag against the pre-tag: V's and W's three-run ranges (overlap or apart) and the change of the
mean V - W. Summary: meshes with mean V - W > 0, meshes where all seeds agree, the paired Wilcoxon test over the meshes'
three-seed means, the median floor-gap share; the same for the pre-tag gallery; meshes where the tag's V range lies
apart from the pre-tag V range (above / below), and the same for W. Writes OUT_MD and OUT_MD's .json."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
args = sys.argv[1:]
opt = {}
for k in ("--dir", "--pre"):
    if k in args:
        i = args.index(k)
        opt[k] = args[i + 1]
        del args[i:i + 2]
DT, DP = G / opt.get("--dir", "d145t_b3"), G / opt.get("--pre", "d145_b3")
SEEDS = (97, 99, 101)
MESHES = "A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot teapot V".split()


def ind(d):
    out = {}
    for s in open(d / "ind.log"):
        if s.startswith("{"):
            r = json.loads(s)
            out[r["tag"]] = r
    return out


def extra(d, tag):
    out = {}
    try:
        arm = json.loads((d / f"{tag}.json").read_text())["arms"]["render_full_dt_iso_nn"]
        kin = [h.get("kin") for h in arm["history"] if h.get("kin") is not None]
        sh = [h.get("g_share") for h in arm["history"] if h.get("outer_accepted") and h.get("g_share") is not None]
        out.update(kin_last=kin[-1] if kin else None, windows=sum(1 for h in arm["history"] if h.get("outer_accepted")),
                   minutes=arm.get("seconds", 0) / 60, share_min=min(sh) if sh else None,
                   share_max=max(sh) if sh else None)
    except (OSError, KeyError, ValueError):
        pass
    try:
        m = re.search(r"windows 3-10: apart max (\d+)", (d / f"apart_{tag}.log").read_text())
        out["apart"] = int(m.group(1)) if m else None
    except OSError:
        pass
    try:
        rr = [json.loads(s) for s in open(d / f"vis_full_{tag}.jsonl") if s.startswith("{")]
        rr = [x for x in rr if x.get("window") == "range"]
        if rr:
            out["sb_run"] = rr[-1]["side_blob"]
    except OSError:
        pass
    return out


def gallery(d):
    I = ind(d)
    rows, means, shares, agree = {}, [], [], 0
    for mesh in MESHES:
        a = {s: I.get(f"{mesh}40k_V{s}") for s in SEEDS}
        b = {s: I.get(f"{mesh}40k_W{s}") for s in SEEDS}
        av = [a[s]["ind"]["sil"] for s in SEEDS if a[s]]
        bv = [b[s]["ind"]["sil"] for s in SEEDS if b[s]]
        fl = [r["floor"]["sil"] for r in list(a.values()) + list(b.values()) if r]
        row = dict(V=av, W=bv, floor=float(np.mean(fl)) if fl else None,
                   chV=[a[s]["ind"]["chamfer"] for s in SEEDS if a[s]], chW=[b[s]["ind"]["chamfer"] for s in SEEDS if b[s]],
                   exV={s: extra(d, f"{mesh}40k_V{s}") for s in SEEDS}, exW={s: extra(d, f"{mesh}40k_W{s}") for s in SEEDS})
        paired = [a[s]["ind"]["sil"] - b[s]["ind"]["sil"] for s in SEEDS if a[s] and b[s]]
        row["diff"] = paired
        if len(paired) == 3:
            row["diff_mean"] = float(np.mean(paired))
            row["agree"] = all(x > 0 for x in paired) or all(x < 0 for x in paired)
            means.append(row["diff_mean"])
            agree += row["agree"] and row["diff_mean"] > 0
            if row["floor"] is not None and row["floor"] - np.mean(bv) > 0:
                row["share"] = row["diff_mean"] / (row["floor"] - np.mean(bv))
                shares.append(row["share"])
        rows[mesh] = row
    summ = dict(n=len(means), positive=sum(m > 0 for m in means), agree_positive=agree,
                median_share=float(np.median(shares)) if shares else None,
                wilcoxon_p=float(wilcoxon(means).pvalue) if len(means) >= 6 else None,
                median_diff=float(np.median(means)) if means else None)
    return rows, summ


rt, st = gallery(DT)
rp, sp = gallery(DP)
f = lambda v: "-" if not v else f"{min(v):.4f} / {np.median(v):.4f} / {max(v):.4f}"
ov = lambda x, y: "-" if not (x and y) else ("overlap" if min(x) <= max(y) and min(y) <= max(x) else
                                              ("**above**" if min(x) > max(y) else "**below**"))
L = [f"### B3 on the tag ({DT.name}): render arm V against its twin W, independent-sample silhouette IoU", "",
     "| mesh | V min / med / max | W min / med / max | V - W per seed | mean | share of floor gap | tag V vs pre-tag V | "
     "tag W vs pre-tag W | mean V - W pre-tag |", "|---|---|---|---|---|---|---|---|---|"]
cnt = {"V": {"above": 0, "below": 0}, "W": {"above": 0, "below": 0}}
for mesh in MESHES:
    r, q = rt[mesh], rp[mesh]
    oV, oW = ov(r["V"], q["V"]), ov(r["W"], q["W"])
    for k, o in (("V", oV), ("W", oW)):
        if "above" in o:
            cnt[k]["above"] += 1
        elif "below" in o:
            cnt[k]["below"] += 1
    L.append(f"| {mesh} | {f(r['V'])} | {f(r['W'])} | " + ", ".join(f"{x:+.4f}" for x in r["diff"]) +
             f" | {r.get('diff_mean', float('nan')):+.4f} | {r.get('share', float('nan')):.2f} | {oV} | {oW} | "
             f"{q.get('diff_mean', float('nan')):+.4f} |")
L += ["", f"Tag: V ahead at the three-seed mean on {st['positive']} of {st['n']}; all seeds agree (positive) on "
      f"{st['agree_positive']}; paired Wilcoxon p = {st['wilcoxon_p']:.2g}; median V - W {st['median_diff']:+.4f}; median "
      f"floor-gap share {st['median_share']:.2f}.",
      f"Pre-tag ({DP.name}): {sp['positive']} of {sp['n']}; all agree {sp['agree_positive']}; p = {sp['wilcoxon_p']:.2g}; "
      f"median V - W {sp['median_diff']:+.4f}; median share {sp['median_share']:.2f}.",
      f"Tag against pre-tag (three-run ranges): V above on {cnt['V']['above']}, below on {cnt['V']['below']}; W above on "
      f"{cnt['W']['above']}, below on {cnt['W']['below']} (of 19).", ""]
# beside: chamfer, at rest, apart, visible pieces, cost (medians over seeds)
L += ["| mesh | chamfer V / W | end kin V / W | apart w3-10 V / W | side blobs run V / W | windows V / W | "
      "minutes V / W | g_share V (min-max) |", "|---|---|---|---|---|---|---|---|"]
md = lambda xs: "-" if not xs else f"{np.median(xs):.3g}"
for mesh in MESHES:
    r = rt[mesh]
    g = lambda ex, k: [e[k] for e in ex.values() if e.get(k) is not None]
    sh = g(r["exV"], "share_min") + g(r["exV"], "share_max")
    L.append(f"| {mesh} | {md(r['chV'])} / {md(r['chW'])} | {md(g(r['exV'], 'kin_last'))} / {md(g(r['exW'], 'kin_last'))} | "
             f"{md(g(r['exV'], 'apart'))} / {md(g(r['exW'], 'apart'))} | {md(g(r['exV'], 'sb_run'))} / "
             f"{md(g(r['exW'], 'sb_run'))} | {md(g(r['exV'], 'windows'))} / {md(g(r['exW'], 'windows'))} | "
             f"{md(g(r['exV'], 'minutes'))} / {md(g(r['exW'], 'minutes'))} | "
             + ("-" if not sh else f"{min(sh):.3f}-{max(sh):.3f}") + " |")
text = "\n".join(L)
print(text)
Path(args[0]).write_text(text + "\n")
Path(args[0]).with_suffix(".json").write_text(json.dumps(dict(tag=rt, pre=rp, summary_tag=st, summary_pre=sp),
                                                          default=float))
