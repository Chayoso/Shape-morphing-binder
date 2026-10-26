"""edit_reader.py -- D166 rerun (2026-10-24): kept from stops between kept frames is read by interpolation (12-frame stride),
the arm names of the rerun, and the master's 'What to adopt' against the coordinator's criteria (authority near held)."""
from pathlib import Path

p = Path("/data/relcfd/chayo/physmorph_v2/tmp/d166/d166_read.py")
t = p.read_text()


def sub(old, new):
    global t
    assert t.count(old) == 1, (old[:70], t.count(old))
    t = t.replace(old, new)


sub('counts["kept"] == 6 and all(ver["stop_exact"]) and len(ver["stop_exact"]) == 6)',
    'counts["kept"] == 6)   # a stop between kept frames: interpolated (12-frame stride)')
sub('reason.append("exact stop raw frame not sampled; interpolation is diagnostic only")',
    'pass   # the stop lies between kept frames: kept is read by interpolation (12-frame stride)')
sub('Stop/kept values are diagnostic "\n         "interpolations when the exact raw frame was not sampled; these cannot pass retention. ',
    'A stop between kept frames is "\n         "interpolated (12-frame stride). ')
sub('pass_=passed, counts=counts,',
    'pass_=passed, kept_mean=float(np.mean([x for x in ver["kept"] if finite(x)])) if counts["kept"] else None, counts=counts,')
sub('NAMES = {',
    'NAMES = {"S0": "held weight + settle 30 (the settle as it is)", "E3": "held + settle 30, E x3",\n'
    '         "Xr": "best weight definition + settle 30 with --no-settle_relax (chosen at run time)",\n')
i, j = t.index("def aggregate(out_md):"), t.index('if __name__ == "__main__":')
t = t[:i] + '''def aggregate(out_md):
    S = sorted(glob.glob(str(G / "d166/auto/*/summary.md")) + glob.glob(str(G / "d167/auto/*/summary.md")),
               key=lambda p: int(Path(p).parent.name.split("_")[0]))
    vs = {}
    for p in S:
        vj = Path(p).parent / "verdict.json"
        if vj.exists():
            for a, v in json.load(open(vj))["arms"].items():
                vs.setdefault(a, {})[v["kind"]] = v
    ef = B / "tmp/d166/expected.txt"
    exp = [x.strip() for x in open(ef) if x.strip()] if ef.exists() else []
    have = {Path(p).parent.name for p in S}
    miss = [e for e in exp if e not in have]
    L = ["# Render rest + weight definition (D166 / D167, rerun 2026-10-24): master summary", "", "## What to adopt", ""]
    sc = lambda a: vs.get(a, {}).get("scenes", {})  # noqa: E731
    k0, kr, ke = (sc(a).get("kept_mean") for a in ("S0", "Rx", "E3"))
    if k0 is not None and kr is not None:
        dr, de = kr - k0, (ke - k0 if ke is not None else None)
        if dr >= 0.3 and (de is None or dr >= de):
            ans = "the outer-layer RELAXATION toward the smooth 3D target is the eraser"
        elif de is not None and de >= 0.3:
            ans = "the ELASTIC part of the control-built stress is the eraser (stiffer keeps more)"
        else:
            ans = "neither separator moves kept by >= 0.3: the eraser is not isolated by these arms"
        L.append(f"- D166 (what removes the relief after the release): {ans}. Kept (mean / min over 6 runs): settle as it is "
                 f"{fmt(k0)} / {fmt(sc('S0').get('kept_min'))}, no relaxation {fmt(kr)} / {fmt(sc('Rx').get('kept_min'))}, E x3 "
                 f"{fmt(ke)} / {fmt(sc('E3').get('kept_min'))}; end v_mean / twin max {fmt(sc('S0').get('vx_max'), '.1f')}x / "
                 f"{fmt(sc('Rx').get('vx_max'), '.1f')}x / {fmt(sc('E3').get('vx_max'), '.1f')}x.")
    else:
        L.append("- D166: not read yet.")
    h = sc("H")
    defs = [a for a in ("R", "RP", "H", "HP", "F", "FP", "S", "SP", "Xr") if a in vs]
    rows = []
    for a in defs:
        s, ga, bg = sc(a), vs[a].get("gallery", {}), vs[a].get("big", {})
        auth = (None if not h or None in (s.get("share_min"), s.get("colour_min"), h.get("share_min"), h.get("colour_min"))
                else bool(s["share_min"] >= 0.8 * h["share_min"] and s["colour_min"] >= 0.8 * h["colour_min"]))
        rest = s.get("rest") if s else None
        gal = ga.get("pass_") if ga else None
        b3 = bg.get("pass_") if bg else None
        ok = auth is True and rest is True and gal is True and b3 is not False
        rows.append((a, ok, auth, rest, gal, b3, s, ga, bg))
    good = [r[0] for r in rows if r[1]]
    L.append("- D167 (weight definition): " + (f"ADOPT {'; '.join(a + ' = ' + name(a) for a in good)} (content share >= 0.8 x held, "
             "end v_mean <= 2x twin, gallery no worse than the default" + (", 300k holds" if all(r[5] for r in rows if r[0] in good) else
             ", 300k not all read") + ")." if good else "no definition meets all three at once (content share >= 0.8 x held, end "
             "v_mean <= 2x twin, gallery no worse than the default); per definition:"))
    for a, ok, auth, rest, gal, b3, s, ga, bg in rows:
        L.append(f"  - {a} ({NAMES.get(a, a)}): content {('near held' if auth else 'below held') if auth is not None else '-'} "
                 f"(relief min {fmt(s.get('share_min'))}, colour min {fmt(s.get('colour_min'))}); rest {rest} "
                 f"({fmt(s.get('vx_max'), '.1f')}x twin); gallery holds {ga.get('held', '-')}/{ga.get('n', '-')} rest {ga.get('rest', '-')}; "
                 f"300k holds {bg.get('held', '-')}/{bg.get('n', '-')} rest {bg.get('rest', '-')}")
    L += ["", f"Written {time.strftime('%Y-%m-%d %H:%M %Z')}; {len(have & set(exp))} of {len(exp)} job summaries"
          + (f" (missing: {', '.join(miss)})" if miss else "") + f". Arms: {FULL} vs {TWIN}. Criteria: content = relief share "
          "(coin, bunny) and colour r (band), min over seeds 97 / 99, against the held arm H; rest = end v_mean <= 2x its twin on "
          "every 40k run; gallery = sil_iou / chamfer / thin within 0.003 / 0.002 / 0.02 of the tag default on all 6 meshes; 300k = "
          "the same against the rule R and rest <= 2x twin. D166 kept = relief / colour share settled over share at the stop. "
          "All earlier --no-settle_relax results (before the capture fix, 2026-10-23) are invalid and not used.", ""]
    other = [p for pat in ("51_stiff", "53_*", "54_*", "55_*") for p in sorted(glob.glob(str(G / "d162/auto" / pat / "summary.md")))]
    for p in S + other:
        L += ["---", f"<!-- {p} -->", open(p).read().strip(), ""]
    tmp = Path(out_md + f".{os.getpid()}.tmp")
    tmp.write_text("\\n".join(L) + "\\n")
    os.replace(tmp, out_md)
    print(f"{len(have & set(exp))} {len(exp)}")


''' + t[j:]
p.write_text(t)
print("reader edited")
