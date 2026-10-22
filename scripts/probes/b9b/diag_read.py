"""diag_read.py DIAG_JSON -- B9b step 1b reader (CPU): per measured window and per variant (the integrator as it is;
the F smoothing off in the continuations), by chain length k: the read p99 and the share of the frozen read removed,
L's p99, the loaded continuation's motion by kind (|v_c|, |omega| R, the deformation's RMS speed; wu/s), its centroid's
drop (wu), its elastic stretch p99 and largest F change over the last window, and the free continuation's RMS speed."""
import json
import sys

d = json.load(open(sys.argv[1]))
for r in d["rows"]:
    print(f"window {r['window']} (t {r['t_start']:.3f} s): frozen p50/p90/p99 " + "/".join(
        f"{r['frozen'][q]:.4f}" for q in ("p50", "p90", "p99")) + "; one-window " + "/".join(
        f"{r['one'][q]:.4f}" for q in ("p50", "p90", "p99")))
    for name in ("as_is", "smoothing_0"):
        print(f"  {name}: k | read p50/p90/p99 | removed p99 | L p99 | loaded v_c / omega R / v_def | drop wu | "
              "loaded strain p99 | dF loaded | free v_rms")
        for c in r[name]:
            m = c["loaded"]
            print(f"    {c['k']} | {c['read']['p50']:.4f}/{c['read']['p90']:.4f}/{c['read']['p99']:.4f} | "
                  f"{100 * (1 - c['read']['p99'] / r['frozen']['p99']):+.0f} % | {c['L']['p99']:.4f} | "
                  f"{m['v_c']:.2e} / {m['omega_R']:.2e} / {m['v_def_rms']:.2e} | {m['drop_wu']:.4f} | "
                  f"{c['loaded_strain']['p99']:.4f} | {c['dF_loaded']:.2e} | {c['free']['v_rms']:.2e}")
