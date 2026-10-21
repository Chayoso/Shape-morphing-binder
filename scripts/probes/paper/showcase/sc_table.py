"""sc_table.py OUT_PREFIX NAME[:TWIN] [...] -- SHOWCASE: the table of the showcase runs (CPU only, reads logs).

Per run NAME in output/gpu/showcase (a name b1_MESH300k_L97 reads D145's row of d145t_b1/MESH300k_L97 from
d145t_evidence/tab_tx.jsonl instead): tab145.py's row (the run's JSON, yardstick t3end_, display e3end_, apart_, m_),
the end check (endgen.jsonl: silhouette IoU and chamfer against the independent and the own sample), the visible blobs
(vis_full_: over the run and in its last window), the pieces off the body at the end (pieces_), the progress-0.5 time
(mid/), the run's wall and peak memory (rc_), the cell (the run log's ppc). With :TWIN, the lead of NAME over its twin
(L - P) / P on the yardstick silhouette, the display front 1 - IoU, 1 - silhouette IoU (independent) and thin.
Writes OUT_PREFIX.jsonl (one row per run) and OUT_PREFIX.md (the table)."""
import json
import re
import subprocess
import sys
from pathlib import Path

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
D = G / "showcase"
TAB = "/data/relcfd/chayo/physmorph_v2/tmp/d145/tab145.py"


def jl(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


def base_row(name):
    if name.startswith("b1_"):
        run = name[3:]
        rows = [r for r in jl(G / "d145t_evidence" / "tab_tx.jsonl") if r.get("dir") == "d145t_b1" and r.get("name") == run]
        return dict(rows[-1]) if rows else {"name": run}
    tmp = D / f"tab_{name}.jsonl"
    tmp.unlink(missing_ok=True)
    subprocess.run([sys.executable, TAB, str(tmp), f"showcase/{name}"], capture_output=True)
    rows = jl(tmp)
    return rows[-1] if rows else {"name": name}


def row(name):
    r = base_row(name)
    r["example"] = name
    src = (D / f"title_{name}").read_text().strip() if (D / f"title_{name}").exists() else ""
    r["title"] = src
    if not name.startswith("b1_"):
        eg = [e for e in jl(D / "endgen.jsonl") if e.get("run") == name]
        if eg:
            e = eg[-1]
            r.update(sil_ind=e["ind"]["sil"], chamfer_ind=e["ind"]["chamfer"], floor_sil=e["floor"])
        vf = jl(D / f"vis_full_{name}.jsonl")
        rng = [v for v in vf if v.get("window") == "range"]
        win = [v for v in vf if v.get("window") != "range"]
        if rng:
            r.update(vis_run_blob=rng[-1]["blob"], vis_run_side_blob=rng[-1]["side_blob"], vis_run_apart=rng[-1]["apart"])
        if win:
            r.update(vis_end_blob=win[-1]["blob"], vis_end_window=win[-1]["window"])
        pc = jl(D / f"pieces_{name}.jsonl")
        if pc:
            p = pc[-1]
            r.update(pc_end_pieces=p["end_pieces"], pc_end_particles=p["end_particles"], pc_end_largest=p["end_largest"],
                     pc_end_ge10=p["end_ge10"], pc_end_ge5a=p["end_ge5a"], pc_end_farthest_a=p["end_farthest_a"])
        try:
            rc = (D / f"rc_{name}").read_text().split()
            r.update(rc=int(rc[0]), wall_s=int(rc[1]), peak_mib=int(rc[2]))
        except (OSError, ValueError, IndexError):
            pass
        try:
            log = (D / f"{name}.log").read_text()
            m = re.search(r"\[disc\] cell from the shape: dx = diag ([\d.]+) / 26 = ([\d.]+) wu -> ppc = N dx\^3 / V = ([\d.]+)", log)
            if m:
                r.update(diag=float(m.group(1)), dx=float(m.group(2)), ppc=float(m.group(3)))
        except OSError:
            pass
    try:
        r["t_mid"] = json.load(open(D / "mid" / f"{name}.json"))["t_mid"]
    except (OSError, KeyError, ValueError):
        pass
    return r


def lead(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


rows, out = [], []
for spec in sys.argv[2:]:
    name, twin = (spec.split(":") + [None])[:2]
    r = row(name)
    if twin:
        t = row(twin)
        r["twin"] = twin
        r["lead_yard_sil"] = lead(r.get("yard_ext_sil"), t.get("yard_ext_sil"))
        r["lead_disp_front"] = lead(r.get("disp_front_1miou"), t.get("disp_front_1miou"))
        r["lead_1miou_ind"] = lead(1 - r["sil_ind"] if r.get("sil_ind") else None, 1 - t["sil_ind"] if t.get("sil_ind") else None)
        r["lead_thin"] = lead(r.get("thin"), t.get("thin"))
        r["twin_row"] = t
    rows.append(r)

with open(f"{sys.argv[1]}.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")


def fmt(v, spec):
    return "–" if v is None else format(v, spec)


def pct(v):
    return "–" if v is None else f"{100 * v:+.0f} %"


hdr = ("| example | sil. IoU (indep.) | yardstick sil. | display 1−IoU front | thin | rest (end kin.) | momentum (COM move a / ang. net:gross) | pieces at end "
       "(all / ≥10 / ≥5 a) | visible blobs (run / end) | minutes | windows | twin lead (yard / display / 1−IoU / thin) |")
lines = [hdr, "|" + "---|" * 12]
for r in rows:
    tl = (" / ".join(pct(r.get(k)) for k in ("lead_yard_sil", "lead_disp_front", "lead_1miou_ind", "lead_thin"))
          if r.get("twin") else "")
    lines.append("| " + " | ".join([
        r["title"] or r["example"], fmt(r.get("sil_ind"), ".4f"), fmt(r.get("yard_ext_sil"), ".2e"),
        fmt(r.get("disp_front_1miou"), ".4f"), fmt(r.get("thin"), ".3f"), fmt(r.get("kin_last"), ".1e"),
        f"{fmt(r.get('com_largest_a'), '.1e')} / {fmt(r.get('ang_net_gross_largest'), '.1e')}",
        f"{r.get('pc_end_pieces', '–')} / {r.get('pc_end_ge10', '–')} / {r.get('pc_end_ge5a', '–')}",
        f"{r.get('vis_run_blob', '–')} / {r.get('vis_end_blob', '–')}",
        fmt(r.get("minutes"), ".1f"), str(r.get("windows", "–")), tl]) + " |")
Path(f"{sys.argv[1]}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
