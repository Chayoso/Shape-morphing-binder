"""d174_memtab.py AUTO_DIR -- D174 memory track: the table of the memory probes (mem_scl{300,500}k[_m1].json): per code, the
device's peak in the window loop, torch's allocated peak, Warp's pool, the persistent buffers by owner (bytes a particle),
and the two-point linear fit used to extrapolate to 1M / 1.5M / 2M / 3M particles."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
G = 2 ** 30


def load(tag):
    out = {}
    for k in (300, 500):
        p = D / f"mem_scl{k}k{tag}.json"
        if p.exists():
            out[k * 1000] = json.load(open(p))
    return out


def stats(rec):
    ph = rec["phases"]
    loop = [r for r in ph if r["label"].startswith("w")]
    w = rec["walks"][0] if rec["walks"] else {"owners": {}}
    own = {}
    for k, v in w["owners"].items():
        name, kind = k.split("|")
        own.setdefault(name, 0)
        own[name] += v[0]
    return dict(device=max(r["device_used"] for r in loop), torch_peak=max(r["torch_peak"] for r in loop),
                warp=max(r["warp_cur"] for r in loop), cupy=max(r["cupy_total"] for r in loop), owners=own,
                walk_total=sum(own.values()))


rows = []
for tag, name in (("", "repo_d174 (batches 1-2)"), ("_m1", "repo_d174m (+ memory batch m1)"), ("_m2", "repo_d174m2 (+ m1 + m2)")):
    R = load(tag)
    if len(R) < 1:
        continue
    S = {n: stats(r) for n, r in R.items()}
    print(f"## {name}")
    print("| N | device peak in the loop | torch peak | Warp pool | CuPy pool | persistent buffers (walk) |")
    print("|---|---|---|---|---|---|")
    for n, s in sorted(S.items()):
        print(f"| {n // 1000}k | {s['device'] / G:.1f} GiB | {s['torch_peak'] / G:.1f} GiB | {s['warp'] / G:.1f} GiB | "
              f"{s['cupy'] / G:.2f} GiB | {s['walk_total'] / G:.1f} GiB |")
    if len(S) == 2:
        (n1, a), (n2, b) = sorted(S.items())
        print()
        print("| owner | B / particle (300k) | B / particle (500k) |")
        print("|---|---|---|")
        for o in sorted(a["owners"], key=lambda o: -a["owners"][o]):
            print(f"| {o} | {a['owners'][o] / n1:.0f} | {b['owners'].get(o, 0) / n2:.0f} |")
        slope = (b["device"] - a["device"]) / (n2 - n1)
        icpt = a["device"] - slope * n1
        print()
        print(f"Linear fit of the device peak: {icpt / G:.2f} GiB + {slope * 1e6 / G:.2f} GiB per million particles")
        print("| N | device peak (extrapolated) |")
        print("|---|---|")
        for n in (1_000_000, 1_500_000, 2_000_000, 3_000_000):
            print(f"| {n / 1e6:g}M | {(icpt + slope * n) / G:.0f} GiB |")
    print()
