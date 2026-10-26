from pathlib import Path
import ast,hashlib,difflib
B=Path('/data/relcfd/chayo/physmorph_v2');P=B/'tmp/d166/d166_read.py';D=B/'tmp/d166/code_audit_20261023'
old=P.read_text();s=old
start=s.index('def run_rec(prefix):');end=s.index('\n\ndef gl():',start)
s=s[:start]+'''def finite(x):
    return x is not None and bool(np.isfinite(x))


def run_rec(prefix):
    """Require successful execution and the exact delivered committed record."""
    p = Path(f"{prefix}.json")
    status = [q for q in (Path(f"{prefix}.rc"), p.parent / f"rc_{p.stem}") if q.exists()]
    if not p.exists() or not status:
        return None
    try:
        if any(int(q.read_text().strip()) != 0 for q in status):
            return None
        a = next(iter(json.load(open(p))["arms"].values()))
        com = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
        if not com:
            return None
        dn = a.get("deliver_n", com[-1]["frame_end"])
        end = next((h for h in com if h["frame_end"] == dn), None)
        com = [h for h in com if h["frame_end"] <= dn]
        opt = [h for h in com if not h.get("settle")]
        m = a.get("metrics", {})
        if end is None or not opt or not all(finite(x) for x in
                (dn, end.get("v_mean"), m.get("sil_iou"), m.get("chamfer"), m.get("thin_uncovered"))):
            return None
        return dict(v_end=end["v_mean"], stop_fe=opt[-1]["frame_end"], deliver_n=dn,
                    settled=any(h.get("settle") for h in com), windows=len(opt),
                    sil=m["sil_iou"], ch=m["chamfer"], thin=m["thin_uncovered"],
                    lam=opt[-1].get("lambda"), sec=a.get("seconds"))
    except (ValueError, TypeError, KeyError, StopIteration, OSError):
        return None


def read_json(path):
    try:
        return json.load(open(path))
    except (ValueError, OSError):
        return {}


def sample_at(xs, ys, x):
    """Return a diagnostic interpolation plus whether x was actually sampled; never extrapolate."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if xs.ndim != 1 or ys.shape != xs.shape or not len(xs) or not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return None, False
    if np.any(np.diff(xs) <= 0) or not finite(x) or x < xs[0] - 1e-8 or x > xs[-1] + 1e-8:
        return None, False
    hit = np.flatnonzero(np.isclose(xs, x, rtol=0, atol=1e-8))
    return (float(ys[hit[0]]), True) if len(hit) else (float(np.interp(x, xs, ys)), False)


def scene_verdict(ver):
    counts = {k: sum(finite(x) for x in ver[k]) for k in ("share", "colour", "vx", "sil", "kept")}
    end_complete = counts["share"] == 4 and counts["colour"] == 2 and counts["vx"] == counts["sil"] == 6
    exact_complete = not ver["settled"] or (counts["kept"] == 6 and all(ver["stop_exact"]) and len(ver["stop_exact"]) == 6)
    complete = bool(end_complete and exact_complete)
    sh = min((x for x in ver["share"] if finite(x)), default=None)
    co = min((x for x in ver["colour"] if finite(x)), default=None)
    vmax = max((x for x in ver["vx"] if finite(x)), default=None)
    kmin = min((x for x in ver["kept"] if finite(x)), default=None) if ver["settled"] else None
    ok_auth = counts["share"] == 4 and counts["colour"] == 2 and sh >= AUTH_RELIEF and co >= AUTH_COLOUR
    ok_rest = counts["vx"] == 6 and vmax <= REST_X
    ok_keep = bool(exact_complete and kmin is not None and kmin >= KEEP) if ver["settled"] else None
    passed = bool(complete and ok_auth and ok_rest and (ok_keep is not False))
    return dict(kind="scenes", reader_schema=2, complete=complete, status="INCOMPLETE" if not complete else ("PASS" if passed else "FAIL"),
                authority=bool(ok_auth), rest=bool(ok_rest), kept=ok_keep, share_min=sh, colour_min=co,
                vx_max=vmax, kept_min=kmin, sil_mean=float(np.mean(ver["sil"])) if counts["sil"] == 6 else None,
                pass_=passed, counts=counts, issues=ver["issues"], diagnostics=ver["diagnostics"])
''' +s[end:]
s=s.replace('if not runs or (D / f"read_{sc}.json").exists():','if not runs or (D / f"read_{sc}.json").exists() or os.environ.get("D166_CACHE_ONLY") == "1":')
s=s.replace('if not (D / "read_colour.jsonl").exists() or (D / "read_colour.jsonl").stat().st_size == 0:  # retry failed empty reader output',
            'if os.environ.get("D166_CACHE_ONLY") != "1" and (not (D / "read_colour.jsonl").exists() or (D / "read_colour.jsonl").stat().st_size == 0):  # retry failed empty reader output')
start=s.index('    rd = {sc: json.load',s.index('def scenes('));end=s.index('    L = [f"# {title}"',start)
s=s[:start]+'''    rd = {sc: read_json(D / f"read_{sc}.json") for sc in ("coin", "bunny")}
    col = {}
    try:
        for line in open(D / "read_colour.jsonl"):
            if line.strip():
                row = json.loads(line)
                col[row["name"]] = row
    except (OSError, ValueError, KeyError):
        col = {}  # An incomplete/corrupt cache cannot support a verdict.
    rows = ["| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |",
            "|---|---|---|---|---|---|---|---|---|"]
    V = {}
    for a, tw in arms:
        ver = dict(share=[], colour=[], vx=[], kept=[], settled=False, sil=[], stop_exact=[], issues=[], diagnostics=[])
        for sc in ("coin", "bunny", "band"):
            st, en, kp, vx, sil, ch, th = [], [], [], [], [], [], []
            for seed in SEEDS:
                r = run_rec(f"{rundir}/{sc}_{a}_{seed}")
                tp = twin_prefix(rundir, sc, tw, seed, "scenes")
                t = run_rec(tp)
                a_stop = a_end = None
                exact_stop = False
                reason = []
                if r is None or t is None:
                    reason.append("missing/failed/nonfinite run or exact delivered record")
                if r is not None:
                    stop_raw, end_raw = r["stop_fe"] - 1, r["deliver_n"] - 1
                    try:
                        if sc != "band":
                            refs = [v["relief"]["A"] for k, v in rd[sc]["refs"].items() if k.startswith("relief_s")]
                            if not refs or not all(finite(v) and v > 0 for v in refs):
                                raise ValueError("missing/nonfinite relief ceiling")
                            ser = rd[sc]["runs"][f"{a}{seed}"]["series"]
                            raws = [q["raw"] for q in ser]
                            vals = np.array([q["relief"]["A"] for q in ser]) / np.mean(refs)
                            a_stop, exact_stop = sample_at(raws, vals, stop_raw)
                            a_end, exact_end = sample_at(raws, vals, end_raw)
                            if not exact_end or not raws or raws[-1] != end_raw:
                                a_end = None
                                reason.append("relief cache delivered frame mismatch")
                        else:
                            c, ct = col[f"band_{a}_{seed}"], col[Path(tp).name]
                            mt, fl, mis = ct["end"]["mis"], c["floor"], c["end"]["mis"]
                            if not all(finite(v) for v in (mt, fl, mis)) or mt - fl <= 1e-9:
                                raise ValueError("nonfinite/degenerate colour reference")
                            if c["deliver_n"] != r["deliver_n"] or ct["deliver_n"] != t["deliver_n"]:
                                raise ValueError("colour cache delivered count mismatch")
                            if abs(c["t_end"] * 240 - end_raw) > 1e-6:
                                raise ValueError("colour cache delivered frame mismatch")
                            rr = lambda m: (mt - m) / (mt - fl)
                            tl = np.asarray(c.get("timeline", []), np.float64)
                            if tl.ndim == 2 and tl.shape[1] == 2:
                                stop_mis, exact_stop = sample_at(tl[:, 0] * 240, tl[:, 1], stop_raw)
                                a_stop = rr(stop_mis) if stop_mis is not None else None
                            a_end = rr(mis)
                            if stop_raw == end_raw:
                                a_stop, exact_stop = a_end, True
                    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
                        a_stop = a_end = None
                        exact_stop = False
                        reason.append("missing/empty/invalid metric cache")
                if not finite(a_stop):
                    a_stop = None
                if not finite(a_end):
                    a_end = None
                    reason.append("end authority unavailable")
                k = a_end / a_stop if a_stop is not None and a_end is not None and a_stop > 0.05 else None
                ratio = r["v_end"] / t["v_end"] if r and t and t["v_end"] > 0 else None
                if r and r["settled"] and not exact_stop:
                    reason.append("exact stop raw frame not sampled; interpolation is diagnostic only")
                st.append(a_stop); en.append(a_end); kp.append(k); vx.append(ratio)
                sil.append(r["sil"] if r else None); ch.append(r["ch"] if r else None); th.append(r["thin"] if r else None)
                ver["settled"] |= bool(r and r["settled"])
                ver["stop_exact"].append(exact_stop)
                ver["issues"] += [f"{sc}/{seed}: {msg}" for msg in reason]
                ver["diagnostics"].append(dict(scene=sc, seed=seed, stop_raw=r["stop_fe"] - 1 if r else None,
                    end_raw=r["deliver_n"] - 1 if r else None, stop_share=a_stop, end_share=a_end,
                    stop_exact=exact_stop, kept_interpolated=k))
            ver["colour" if sc == "band" else "share"] += en
            ver["kept"] += kp
            ver["vx"] += vx
            ver["sil"] += sil
            rows.append(f"| {a} | {sc} | {cell(st, '.2f')} | {cell(en, '.2f')} | {cell(kp, '.2f')} | {cell(vx, '.1f')}x | "
                        f"{cell(sil, '.4f')} | {cell(ch, '.4f')} | {cell(th, '.3f')} |")
        V[a] = scene_verdict(ver)
''' +s[end:]
s=s.replace('"optimised window, end = delivered (after settling if settled); kept = end / stop. Criteria: authority = relief >= "',
'''"optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic "
         "interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, "
         "failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= "''')
s=s.replace("f\"{a} {'PASS' if v['pass_'] else 'FAIL'} (authority", "f\"{a} {v['status']} (authority")
s=s.replace('    open(D / "summary.md", "w").write("\\n".join(L) + "\\n")\n    json.dump(dict(title=title, arms=V)', '    L += ["", "Incomplete evidence:"] + [f"- {a}: {msg}" for a, v in V.items() for msg in v.get("issues", [])]\n    open(D / "summary.md", "w").write("\\n".join(L) + "\\n")\n    json.dump(dict(title=title, arms=V)',1)
s=s.replace('good = [a for a, v in p1.items() if v["pass_"]]','good = [a for a, v in p1.items() if v.get("complete", False) and v["pass_"]]')
s=s.replace('if v.get("scenes", {}).get("pass_") and v.get("gallery", {}).get("pass_", False)', 'if v.get("scenes", {}).get("complete", False) and v.get("scenes", {}).get("pass_") and v.get("gallery", {}).get("pass_", False)')
s=s.replace('and v.get("big", {}).get("pass_", True)]','and v.get("big", {}).get("pass_", False)]')
s=s.replace('if ok else "no definition meets authority + rest + gallery; per definition:"','if ok else "no definition has complete passing authority + rest + gallery + 300k evidence; per definition:"')
s=s.replace('    tmp = Path(out_md + ".tmp")','    tmp = Path(out_md + f".{os.getpid()}.tmp")')
assert s!=old and 'mesh=bunny' in s
ast.parse(s)
# No concurrent edits may be overwritten.
assert P.read_text()==old
P.write_text(s)
(D/'reader_strict.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='d166_read.before_strict.py',tofile='d166_read.py')))
print('patched',hashlib.sha256(P.read_bytes()).hexdigest())
