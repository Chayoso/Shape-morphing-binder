"""d162_colour.py OUT_MD JSONL [JSONL ...] -- D162's colour table (CPU) from d160_read.py's lines: per scene (band, ears)
and seed, the mis-coloured share of the twin P / TC, the tag's rule L, the held weight H (D160c), D162's F and FC, the
recovered share r = (mis_twin - mis_arm) / (mis_twin - floor) (F against D160's P, FC against its own twin TC, L and H
against P), and the held-out colour PSNR / LPIPS of each arm against its twin. C1 per scene: r >= 0.5, PSNR above and
LPIPS below the twin's at both seeds."""
import json
import sys

rows = {}
for f in sys.argv[2:]:
    for ln in open(f):
        r = json.loads(ln)
        rows[r["name"]] = r
out = ["| scene | seed | arm | mis (twin) | r | held PSNR arm / twin | held LPIPS arm / twin | C1 |", "|" + "---|" * 8]
verdict = {}
for sc in ("band", "ears"):
    for s in (97, 99):
        P = rows.get(f"bunny40k_{sc}_P{s}")
        TC = rows.get(f"bunny40k_{sc}_TC_{s}")
        for arm, name, twin in (("L (rule)", f"bunny40k_{sc}_L{s}", P), ("H (held)", f"bunny40k_{sc}_H{s}", P),
                                ("F", f"bunny40k_{sc}_F_{s}", P), ("FC", f"bunny40k_{sc}_FC_{s}", TC)):
            a = rows.get(name)
            if a is None or twin is None:
                continue
            mt, ma, fl = twin["end"]["mis"], a["end"]["mis"], twin["floor"]
            r = (mt - ma) / max(mt - fl, 1e-9)
            pa, pt = a["img"]["held"]["colour"]["psnr"], twin["img"]["held"]["colour"]["psnr"]
            la, lt = a["img"]["held"]["colour"]["lpips"], twin["img"]["held"]["colour"]["lpips"]
            ok = r >= 0.5 and pa > pt and la < lt
            verdict.setdefault((sc, arm), []).append(ok)
            out.append(f"| {sc} | {s} | {arm} | {ma:.3f} ({mt:.3f}, floor {fl:.3f}) | {r:.2f} | {pa:.2f} / {pt:.2f} | "
                       f"{la:.3f} / {lt:.3f} | {'yes' if ok else 'no'} |")
out.append("")
for (sc, arm), v in verdict.items():
    out.append(f"{sc} {arm}: C1 at both seeds: {'PASS' if len(v) == 2 and all(v) else 'no'}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("\n".join(out))
