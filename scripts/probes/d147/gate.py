"""gate.py RUNS_DIR TAGS REFS -- D147, the fixes' 40k gate (pre-registered in docs/experiments.md, D147, 2026-10-16):
per run, (1) the zero-control windows by their reason (nonfinite_loss, commit_replay, commit_jt after the patience:
the causes the fixes remove; the others reported), the windows with a grown raster, a body left out, the replay
retries; (2) the head's hold after each landing (head.py's recovery mean per body) against the reference runs REFS (the
largest of theirs per body); (3) clean: no piece apart from its body of 10 or more particles beyond 3 spacings before
the cut (apart.py over the whole run; the far frames reported); (4) the judge (judge.py); (5) the tumble (reported);
(6) momentum per body: in flight before the first contact each body's P against P0 + M g t, the pair's P and its
grid share of L (gravity's torque out) over the first contact's windows, and each body's grid share of its own L in
free flight between the parting and its landing; (7) the cost. Files read in RUNS_DIR: <tag>.json, judge_gate.txt,
head_gate.txt, cut_<tag>.txt, apart_<tag>.txt, read_gate.txt (read40k.py's timeline). Prints a JSON line per run and
PASS / FAIL. CPU."""
import json
import sys
from pathlib import Path

import numpy as np

CAUSES = ("nonfinite_loss", "commit_replay", "commit_jt")


def lines(path):
    out = {}
    if not Path(path).exists():
        return out
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{") and '"tag"' in ln:
            try:
                j = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.setdefault(j["tag"], {}).update(j)
    return out


runs = Path(sys.argv[1])
tags, refs = sys.argv[2].split(","), sys.argv[3].split(",")
judge, head, tl = lines(runs / "judge_gate.txt"), lines(runs / "head_gate.txt"), lines(runs / "read_gate.txt")
ref_head = [max((b["recovery_mean"] for r in refs for b in head.get(r, {}).get("bodies", []) if b["body"] == k
                 and b.get("recovery_mean") is not None), default=None) for k in range(2)]
print(json.dumps({"refs": refs, "ref_head_recovery_mean_max": ref_head}))
ok_all = True
for tag in tags:
    j = json.loads((runs / f"{tag}.json").read_text())
    h = [r for r in j["history"] if "bodies" in r]
    g = -j["mpm"]["f_ext"][1]
    win = 2 * j["config"]["T"] * j["mpm"]["dt"]
    t_cut = json.loads(open(runs / f"cut_{tag}.txt").readline())["t_cut"]
    # (1)
    zero = {}
    for r in h:
        if r.get("zero_commit"):
            zero.setdefault(r.get("zero_reason") or r.get("null_reason") or "none", []).append(round(r["t"], 2))
    caused = {k: v for k, v in zero.items() if k in CAUSES}
    grown = [[round(r["t"], 2) for r in h if (r.get("raster_k") or [0, 0])[b] > 0] for b in range(2)]
    left = [round(r["t"], 2) for r in h if r.get("inactive")]
    retries = sum(len(r.get("replay_retries") or []) for r in h)
    # (2)
    hb = head.get(tag, {}).get("bodies", [])
    hold = [b.get("recovery_mean") for b in sorted(hb, key=lambda b: b["body"])]
    hold_ok = len(hold) == 2 and all(v is not None and rv is not None and v <= rv for v, rv in zip(hold, ref_head))
    # (3)
    far, big = [], 0
    ap = runs / f"apart_{tag}.txt"
    for ln in open(ap) if ap.exists() else []:
        if ln.startswith('{"t"'):
            p = json.loads(ln)
            if p["t"] <= t_cut + 1e-9 and p["to_body_sp"] > 3.0:
                far.append((p["t"], p["body"], p["n"], p["to_body_sp"]))
                big = max(big, p["n"])
    clean_ok = ap.exists() and big < 10
    # (4), (5)
    jd, t = judge.get(tag, {}), tl.get(tag, {})
    # (6) momentum per body
    hit = [bool(np.linalg.norm(r["contact_impulse"]) > 0) for r in h]
    k0 = hit.index(True) if any(hit) else len(h)
    mass = [float(j["config"]["mass_ref_n"])] * 2                 # each body's mass (the dynamics mass, mass_ref_n)
    P0 = [np.array(b["P"]) for b in h[0]["bodies"]]
    t0 = h[0]["t"]
    flight = max((np.linalg.norm(np.array(r["bodies"][b]["P"]) - P0[b] + np.array([0.0, mass[b] * g * (r["t"] - t0), 0.0]))
                  / max(np.linalg.norm(P0[b]), 1e-9) for r in h[:k0] for b in range(2)), default=0.0)
    k1 = next((k for k in range(k0, len(h)) if not hit[k]), len(h))
    pre = h[k0 - 1] if k0 > 0 else h[0]
    Pa = sum(np.array(b["P"]) for b in pre["bodies"])
    Pb = sum(np.array(b["P"]) for b in h[k1 - 1]["bodies"]) if k1 > k0 else Pa
    dP_pair = np.linalg.norm(Pb - Pa + np.array([0.0, sum(mass) * g * (h[k1 - 1]["t"] - pre["t"]), 0.0]))
    m_p = float(j["config"]["mass_ref_n"]) / (j["provenance"]["scene_args"]["n"])
    sel = [r for r in h[k0:k1] if r.get("L_split")]
    Lp = (m_p * np.sum([[r["L_split"][b]["pair"]["L_grid"] for b in range(2)] for r in sel], axis=(0, 1))
          - m_p * np.sum([[r["L_split"][b]["pair"]["L_gravity"] for b in range(2)] for r in sel], axis=(0, 1))) \
        if sel else np.zeros(3)
    Lscale = max((m_p * np.linalg.norm(r["L_split"][b]["pair"]["L_start"]) for r in sel for b in range(2)), default=1.0)
    land = t.get("landing_s") or [None, None]
    t_part = t.get("t_part")
    own = []
    for b in range(2):
        fs = [r for r in h if t_part is not None and land[b] is not None and t_part < r["t"] <= land[b] - win
              and r.get("L_split")]
        Lg = m_p * np.sum([r["L_split"][b]["own"]["L_grid"] for r in fs], axis=0) if fs else np.zeros(3)
        Ls = max((np.linalg.norm(r["bodies"][b]["L"]) for r in fs), default=1.0)
        own.append(float(np.linalg.norm(Lg) / max(Ls, 1.0)))
    mom = {"flight_P_rel": float(flight), "contact_dP_pair_rel": float(dP_pair / max(np.linalg.norm(P0[0]), 1e-9)),
           "contact_L_pair_rel": float(np.linalg.norm(Lp) / max(Lscale, 1.0)), "free_L_own_rel": own}
    mom_ok = mom["flight_P_rel"] <= 1e-4 and mom["contact_dP_pair_rel"] <= 1e-4 and mom["contact_L_pair_rel"] <= 1e-4 \
        and all(v <= 1e-3 for v in own)
    ok = (not caused) and hold_ok and clean_ok and bool(jd.get("pass")) and mom_ok
    ok_all &= ok
    print(json.dumps({"tag": tag, "pass": ok, "zero_caused": caused, "zero_other": {k: v for k, v in zero.items()
                                                                                       if k not in CAUSES},
                      "raster_grown_t": grown, "left_out_t": left, "replay_retries": retries,
                      "head_recovery_mean": hold, "head_ref_max": ref_head, "head_ok": hold_ok,
                      "head_recovery_max": [b.get("recovery_max") for b in sorted(hb, key=lambda b: b["body"])],
                      "far_frames": len({f[0] for f in far}), "far_particle_frames": int(sum(f[2] for f in far)),
                      "far_largest": big, "far_max_sp": max((f[3] for f in far), default=0.0), "clean_ok": clean_ok,
                      "judge_pass": jd.get("pass"), "end_error": jd.get("end_error"), "progress": jd.get("progress"),
                      "parted_in_air": jd.get("parted_in_air"), "landed": jd.get("landed"),
                      "early_ratio": jd.get("early_ratio"), "tumble_deg": jd.get("tumble_deg"),
                      "momentum": mom, "momentum_ok": mom_ok, "t_cut": t_cut, "minutes": round(j["seconds"] / 60, 1)}))
print("PASS" if ok_all else "FAIL")
