"""t5_145.py HARNESS_DIR ROWS_JSONL MESH [...] -- D145 T5 (methods): per mesh and method, the harness's progress and
intermediate-frame measures (harness_MESH.jsonl: p at the end; apart share and occupied-volume ratio at p = 0.25 /
0.5 / 0.75 and the end; the frame time tau) with the display at those frames (e3h_MESH_LABEL.log: 1 - IoU front /
back, apart discs and sets against the independent sample), and the end rows of tab145.py (yardstick, display,
silhouette IoU / chamfer, at rest, minutes) where the method has them. CPU only; prints a table, writes t5.json."""
import json
import sys
from pathlib import Path

H = Path(sys.argv[1])
rows = {}
try:
    for s in open(sys.argv[2]):
        if s.startswith("{"):
            r = json.loads(s)
            rows[r["name"]] = r
except OSError:
    pass
out = []
for mesh in sys.argv[3:]:
    try:
        hs = [json.loads(s) for s in open(H / f"harness_{mesh}.jsonl") if s.startswith("{")]
    except OSError:
        continue
    for h in hs:
        disp = {}
        try:
            for s in open(H / f"e3h_{mesh}_{h['label']}.log"):
                if s.startswith("{"):
                    r = json.loads(s)
                    if r.get("state") not in ("target", "own"):
                        tt = r.get("to_target", {})
                        disp[str(r["state"])] = dict(front=1 - tt.get("front", {}).get("iou", float("nan")),
                                                     back=1 - tt.get("back", {}).get("iou", float("nan")),
                                                     apart=r.get("apart"), sets=r.get("sets"))
        except OSError:
            pass
        rec = dict(mesh=mesh, label=h["label"], p_end=h["p_end"], escaped_max=h["escaped_max"])
        for key in ("p0.25", "p0.5", "p0.75", "end"):
            v = h["read"].get(key)
            if v:
                rec[key] = dict(tau=v["tau"], raw=v["raw"], apart_share=v["apart_share"], occ=v["occupied_ratio"],
                                display=disp.get(str(v["raw"])))
        out.append(rec)
        f = lambda k: ("-" if k not in rec else f"tau {rec[k]['tau']:.2f} apart {rec[k]['apart_share'] * 100:.2f}% "
                       f"occ {rec[k]['occ']:.3f}" + ("" if not rec[k]["display"] else
                                                     f" disp {rec[k]['display']['front']:.4f} sets {rec[k]['display']['sets']}"))
        print(f"{mesh:9s} {h['label']:6s} p_end {h['p_end']:.4f} | p.25 {f('p0.25')} | p.5 {f('p0.5')} | p.75 {f('p0.75')} | end {f('end')}")
(H / "t5.json").write_text(json.dumps(out))
