"""B12 (D145 amendment 5) conversion, B5's convert_pair.py for ISD / 4Deform: the 21 evaluated level sets of one run
(every component; t = k/20) back to the pipeline frame (x = (y / s + c) / a: prep_pair.py's common centring c and scale
s from pair_info.json, then B5's export scale a), their mesh properties (also of the largest-area component, the
authors' own display), and the pipeline's stratified 300k volume sample (seed 97) of each; written as the archive
output/gpu/d145_b12/<name>300k_<TAG>_frames12.npz (frames (21, 300000, 3), raws, tgt = the pipeline's seed-97 target
sample, deliver_n = 21). A frame with no level set or one the sampler cannot fill is NaN in the archive and named in the
JSON. PLYs kept in the pipeline frame for t in {0.25, 0.5, 0.75, 1} (all 21 for bunny and dragon); the originals deleted.

  $PY convert_b12.py NAME TAG RECON_DIR DATA_DIR
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import trimesh

BU = "/data/relcfd/chayo/physmorph_v2/baselines/buonomo"
R = "/data/relcfd/chayo/physmorph_v2/baselines/b12"
NAME, TAG, RECON, DATA = sys.argv[1:5]
N, SEED, T = 300000, 97, 21
OUT_NPZ = f"/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12/{NAME}300k_{TAG}_frames12.npz"
KEEP_PLY = {5, 10, 15, 20} if NAME not in ("bunny", "dragon") else set(range(T))
exp = json.load(open(f"{BU}/inputs/{NAME}/export.json"))
A = exp["a_to_buonomo"]
TGT_FILL = exp["fill_tgt"]
info = json.load(open(os.path.join(DATA, "pair_info.json")))
S = float(info["scale"])
C = np.asarray(info["center_src"], np.float64)
assert np.allclose(C, np.asarray(info["center_tgt"])), "prep_pair.py --center common expected"


def tgt_signed_volume():
    p = f"{BU}/inputs/{NAME}/{NAME}_outer_ourframe.ply"           # teapot: the outer body (its cavities filled)
    m = trimesh.load(p if os.path.exists(p) else f"{BU}/inputs/{NAME}/{NAME}_ourframe.ply", process=True)
    return float(m.volume)


def comp_rows(comps):
    return [dict(F=int(len(c.faces)), area=float(c.area), watertight=bool(c.is_watertight), euler=int(c.euler_number),
                 genus=(int(round((2 - c.euler_number) / 2)) if c.is_watertight else None),
                 volume=(float(c.volume) if c.is_watertight else None)) for c in comps[:6]]


def props(k):
    from physmorph.sampling.mesh import sample_volume_stratified
    t0 = time.time()
    p = glob.glob(os.path.join(RECON, f"time_{k}.*"))[0]
    raw = trimesh.load(p, process=False, force="mesh")
    r = dict(k=k, t=k / (T - 1), V=int(len(raw.vertices)), F=int(len(raw.faces)))
    nan = np.full((N, 3), np.nan, np.float32)
    if len(raw.faces) == 0:
        r.update(empty=True, components=0, comp=[], volume=None, signed_volume=0.0, sampled=False)
        return r, nan
    vb = np.asarray(raw.vertices, np.float64)
    r["raw_bounds"] = [vb.min(0).tolist(), vb.max(0).tolist()]
    m = trimesh.Trimesh((vb / S + C) / A, np.asarray(raw.faces), process=True)
    r.update(watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
             euler=int(m.euler_number))
    repaired = None
    if not m.is_watertight:
        mr = m.copy()
        trimesh.repair.fill_holes(mr)
        repaired = bool(mr.is_watertight)
        if mr.is_watertight:
            m = mr
    r["fill_holes"] = repaired
    comps = m.split(only_watertight=False)
    by_area = sorted(comps, key=lambda c: -c.area)
    r["largest_area_comp"] = comp_rows(by_area[:1])[0] if by_area else None     # the authors' display
    comps = sorted(comps, key=lambda c: -abs(c.volume) if c.is_watertight else -len(c.faces))
    r["components"] = len(comps)
    r["comp"] = comp_rows(comps)
    r["volume"] = float(m.volume) if m.is_watertight else None
    r["signed_volume"] = float(m.volume)
    try:
        x = sample_volume_stratified(m, N, seed=SEED).astype(np.float32)
        r["sampled"] = True
    except Exception as e:                                          # noqa: BLE001  (recorded as found)
        x = nan
        r["sampled"] = False
        r["sample_error"] = repr(e)
    r["sample_s"] = time.time() - t0
    if k in KEEP_PLY:
        m.export(os.path.join(RECON, f"ourframe_time_{k:02d}.ply"), encoding="binary")
    return r, x


if __name__ == "__main__":
    t0 = time.time()
    with Pool(7) as pool:
        res = pool.map(props, range(T))
    rows = [r for r, _ in res]
    frames = np.stack([x for _, x in res]).astype(np.float32)
    tgt = np.asarray(np.load(f"{BU}/inputs/{NAME}/pipeline_samples.npz")["tgt"], np.float32)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, frames=frames, raws=np.arange(T), tgt=tgt, deliver_n=np.int64(T))
    tv = tgt_signed_volume()
    v0 = rows[0]["signed_volume"] or float("nan")
    for r in rows:
        r["vol_ratio_tgt_mesh"] = r["signed_volume"] / tv
        r["vol_ratio_tgt_fill"] = r["signed_volume"] / TGT_FILL
        r["vol_ratio_t0"] = r["signed_volume"] / v0
    summary = dict(name=NAME, tag=TAG, a=A, scale=S, center=C.tolist(), tgt_mesh_signed_volume=tv, tgt_fill_volume=TGT_FILL,
                   npz=OUT_NPZ, convert_s=time.time() - t0, unsampled=[r["k"] for r in rows if not r.get("sampled")],
                   rows=rows)
    json.dump(summary, open(f"{R}/results/{NAME}_{TAG}_meshes.json", "w"), indent=1)
    for p in glob.glob(os.path.join(RECON, "time_*.*")):
        os.remove(p)                                                # the method-frame originals (kept: ourframe_*)
    for r in rows:
        c0 = r["comp"][0] if r.get("comp") else {}
        la = r.get("largest_area_comp") or {}
        print(f"t={r['t']:.2f} wt={r.get('watertight')} fill_holes={r.get('fill_holes')} comps={r.get('components')} "
              f"genus0={c0.get('genus')} euler={r.get('euler')} largest(area): wt={la.get('watertight')} "
              f"genus={la.get('genus')} vol/tgt={r['vol_ratio_tgt_mesh']:.4f} vol/t0={r['vol_ratio_t0']:.4f} "
              f"sampled={r.get('sampled')} V={r['V']}", flush=True)
    print(f"wrote {OUT_NPZ} in {time.time() - t0:.0f} s", flush=True)
