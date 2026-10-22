"""settle_read.py SETTLE_JSON -- B9b step 1 reader (CPU): the settle profile by chain length k (median over the measured
windows): the read p99 and the share of the frozen read removed, the load effect's size, each continuation's elastic
stretch against Fp, its largest F change over its last window and its RMS speed; the carried pair against the settled."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
rows = d["rows"]
ks = [c["k"] for c in rows[0]["chain"]]
print(f"{len(rows)} windows, t {rows[0]['t_start']:.3f}-{rows[-1]['t_start']:.3f} s; frozen p99 median "
      f"{np.median([r['frozen']['p99'] for r in rows]):.4f}; one-window p99 median {np.median([r['one']['p99'] for r in rows]):.4f}"
      f" (removed {100 * np.median([1 - r['one']['p99'] / r['frozen']['p99'] for r in rows]):.0f} %)")
print("k | read p50/p90/p99 | removed p99 % | L p99 | loaded strain p99 | free strain p99 | dF loaded | dF free | "
      "v_rms loaded | v_rms free (wu/s)")
for i, k in enumerate(ks):
    c = [r["chain"][i] for r in rows]
    med = lambda f: float(np.median([f(x) for x in c]))  # noqa: E731
    rem = float(np.median([100 * (1 - r["chain"][i]["read"]["p99"] / r["frozen"]["p99"]) for r in rows]))
    print(f"{k} | {med(lambda x: x['read']['p50']):.4f}/{med(lambda x: x['read']['p90']):.4f}/{med(lambda x: x['read']['p99']):.4f}"
          f" | {rem:.0f} | {med(lambda x: x['L']['p99']):.4f} | {med(lambda x: x['loaded_strain']['p99']):.4f} | "
          f"{med(lambda x: x['free_strain']['p99']):.4f} | {med(lambda x: x['dF_loaded']):.2e} | {med(lambda x: x['dF_free']):.2e} | "
          f"{med(lambda x: x['v_rms_loaded']):.2e} | {med(lambda x: x['v_rms_free']):.2e}")
print("frozen p50/p90/p99 median: " + "/".join(f"{np.median([r['frozen'][q] for r in rows]):.4f}" for q in ("p50", "p90", "p99")))
print("one-window p50/p90/p99 median: " + "/".join(f"{np.median([r['one'][q] for r in rows]):.4f}" for q in ("p50", "p90", "p99")))
print("carried (window: read p99, settled read p99, L max / rms difference from the settled):")
for r in rows:
    c = r["carried"]
    print(f"  {r['window']}: {c['read']['p99']:.4f} vs {r['chain'][-1]['read']['p99']:.4f}; "
          f"{c.get('L_vs_settled_max', float('nan')):.3e} / {c.get('L_vs_settled_rms', float('nan')):.3e}")
print("first window vs the definition's continuations (largest F difference): "
      + ", ".join(f"{r['first_vs_definition']['loaded']:.1e}/{r['first_vs_definition']['free']:.1e}" for r in rows))
