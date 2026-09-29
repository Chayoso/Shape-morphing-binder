"""gc_read.py JSON [LAMBDA] — print a gradient-check report. LAMBDA overrides lam_r for the influence ratios (the window-1
hook runs before the balancer has set lambda; the run's calibrated lambda is used then)."""
import sys, json
r = json.load(open(sys.argv[1]))
lam = float(sys.argv[2]) if len(sys.argv) > 2 else r["lam_r"]
print(f"window {r['window']}, N {r['N']}, rollout steps {r['rollout_steps']} (controlled {r['control_steps']}), lam_r at the hook "
      f"{r['lam_r']:.4f}, lambda used below {lam:.4f}")
print(f"release-phase dFc |max| {r['release_dfc_absmax']}, driven-phase dFc |max| {r['driven_dfc_absmax']:.3e}")
g = r["graph_vs_eval_at_leaf0"]
print("graph path vs line-search path at the same controls: " + ", ".join(
    f"{k} {g[k + '_graph']:.7f} / {g[k + '_eval']:.7f}" for k in ("phys", "render", "cleanup")))
print("replay noise: " + ", ".join(f"{k} {v:.1e}" for k, v in r["replay_noise"].items()))
print("-- render influence (norms of gradients on the control leaves)")
for k, v in r["render_influence"].items():
    gp, gr = v["|g_phys|"], v["|g_render_proj|"]
    share = lam * gr / max(gp + lam * gr, 1e-30)
    print(f"   {k:6s} |g_phys| {gp:.3e}  |g_render| raw {v['|g_render_raw|']:.3e} proj {gr:.3e}  cos {v['cos_raw']:+.3f}  "
          f"lambda*|g_render|/|g_phys| {lam * gr / max(gp, 1e-30):.3f}  render share {share:.3f}"
          + (f"  PCGrad projected {v['projected']}  |g_cleanup| {v['|g_cleanup|']:.3e}" if k == "total" else ""))
print("-- finite differences through the line-search rollout (fd) vs autograd (ag); ratio fd/ag")
for dn, rows in r["finite_difference"].items():
    for row in rows:
        parts = []
        for k in ("phys", "render", "cleanup"):
            fd, ag = row[k]["fd"], row[k]["autograd"]
            ratio = fd / ag if abs(ag) > 1e-30 else float("nan")
            parts.append(f"{k} fd {fd:+.3e} ag {ag:+.3e} ({ratio:5.2f})")
        print(f"   {dn:11s} h {row['h']:<5} " + "  ".join(parts))
print(f"-- momentum budget of the {r['rollout_steps']}-step rollout (total mass {r['mass_total']:.1f}, dt {r['dt']:.5f}, drag {r['drag']})")
keys = ["|P|/sum m|v| max", "|P|/sum m|v| end", "|L|/sum m|r||v| max", "|L|/sum m|r||v| end", "COM displacement (wu)",
        "COM displacement explained by momentum (wu)", "COM displacement unexplained (wu)", "max COM step (wu)",
        "driven-phase end P/M (wu/s)", "rollout end P/M (wu/s)", "sum m|v| end / max", "sum m|v| max / M (wu/s)"]
short = ["|P|/Σm|v| max", "|P|/Σm|v| end", "|L|/Σm|r||v| max", "|L| end", "ΔCOM", "ΔCOM by P", "ΔCOM unexpl.", "max COM step",
         "P/M drive end", "P/M end", "Σm|v| end/max", "Σm|v| max/M"]
for k, v in r["momentum"].items():
    print(f"   {k}")
    print("      " + "  ".join(f"{s} {v[kk]:.2e}" for s, kk in zip(short, keys) if kk in v))
