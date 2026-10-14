"""tab145.py OUT_JSONL DIR/NAME [...] -- D145's row per 300k run, from the run's own JSON and read300.sh's logs (CPU only).

Per run (output/gpu/DIR/NAME*): the run's own metrics (silhouette IoU, chamfer, thin uncovered against its own target
sample), windows, attempts, minutes, seconds per attempt, the last kinetic record, the render weight and share
(first window, median); the yardstick at the end (t3end_: exterior and particle silhouette / shading against the
independent sample); the display at the end (e3end_: 1 - IoU and difference front / crop / back, sets, apart discs,
field-normal angle rms); apart particles (apart_: windows 3-10 max / median, 10-20 max); momentum (m_: the summary
line); the tail pulse (pulse_, its last lines verbatim). Missing pieces are null. One JSON line per run."""
import json
import re
import sys
from pathlib import Path

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")


def jlines(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


rows = []
for spec in sys.argv[2:]:
    d, name = spec.split("/")
    D = G / d
    row = {"dir": d, "name": name}
    try:
        js = json.loads((D / f"{name}.json").read_text())
        arm = js["arms"]["render_full_dt_iso_nn"]
        met, hist = arm["metrics"], arm["history"]
        committed = [h for h in hist if h.get("outer_accepted")]
        lam = [h.get("lambda") for h in hist if h.get("lambda") is not None]
        share = [h.get("g_share") for h in hist if h.get("g_share") is not None]
        kin = [h.get("kin") for h in hist if h.get("kin") is not None]
        row.update(sil_own=met.get("sil_iou"), chamfer_own=met.get("chamfer"), thin=met.get("thin_uncovered"),
                   thin_world=met.get("thin_uncovered_world"), surf_rough=met.get("surf_rough"),
                   attempts=len(hist), windows=len(committed), deliver_n=arm.get("deliver_n"),
                   minutes=arm.get("seconds", 0) / 60, s_attempt=arm.get("seconds", 0) / max(len(hist), 1),
                   kin_last=kin[-1] if kin else None, lam_first=lam[0] if lam else None,
                   lam_median=sorted(lam)[len(lam) // 2] if lam else None, share_first=share[0] if share else None,
                   share_median=sorted(share)[len(share) // 2] if share else None,
                   guards=arm.get("guards"))
    except (OSError, KeyError, ValueError) as e:
        row["json_error"] = repr(e)
    t3 = jlines(D / f"t3end_{name}.log")
    if t3:
        row.update(yard_ext_sil=t3[-1]["exterior"]["sil"], yard_ext_pbr=t3[-1]["exterior"]["pbr"],
                   yard_par_sil=t3[-1]["particles"]["sil"], yard_par_pbr=t3[-1]["particles"]["pbr"],
                   yard_state=t3[-1]["state"])
    e3 = [r for r in jlines(D / f"e3end_{name}.log") if r.get("state") not in ("target", "own")]
    if e3:
        r = e3[-1]
        tt = r.get("to_target", {})
        row.update({f"disp_{v}_1miou": 1 - tt[v]["iou"] for v in ("front", "crop", "back") if v in tt})
        row.update({f"disp_{v}_diff": tt[v]["difference"] for v in ("front", "crop", "back") if v in tt})
        row.update(disp_sets=r.get("sets"), disp_apart=r.get("apart"),
                   disp_angle_rms=r.get("field_normals", {}).get("angle_rms"), disp_state=r.get("state"))
    try:
        ap = (D / f"apart_{name}.log").read_text()
        row.update(apart_w3_10_max=num(r"windows 3-10: apart max (\d+)", ap, int),
                   apart_w3_10_median=num(r"windows 3-10: apart max \d+ median (\d+)", ap, int),
                   apart_w10_20_max=num(r"windows 10-20: max (\d+)", ap, int))
    except OSError:
        pass
    try:
        mt = [s for s in (D / f"m_{name}.log").read_text().splitlines() if "kept frames |" in s]
        if mt:
            s = mt[-1]
            row.update(com_largest_a=num(r"largest displacement ([\d.e+-]+) a", s),
                       com_end_a=num(r"at the end ([\d.e+-]+) a", s),
                       lin_net_gross_largest=num(r"net / gross linear move: largest ([\d.e+-]+)", s),
                       ang_net_gross_largest=num(r"net / gross angular move: largest ([\d.e+-]+)", s),
                       ang_net_gross_mean=num(r"net / gross angular move: largest [\d.e+-]+, mean ([\d.e+-]+)", s),
                       rotation_deg=num(r"net rotation over the run ([\d.e+-]+) deg", s),
                       move_last20_a=num(r"last 20: ([\d.e+-]+) a", s))
    except OSError:
        pass
    try:
        row["pulse_tail"] = (D / f"pulse_{name}.log").read_text().strip().splitlines()[-3:]
    except OSError:
        pass
    rows.append(row)

with open(sys.argv[1], "a") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")
keys = ["name", "sil_own", "chamfer_own", "thin", "yard_ext_sil", "yard_ext_pbr", "disp_front_1miou", "disp_crop_1miou",
        "kin_last", "apart_w3_10_max", "windows", "minutes"]
print(" | ".join(keys))
for r in rows:
    print(" | ".join("-" if r.get(k) is None else (f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])) for k in keys))
