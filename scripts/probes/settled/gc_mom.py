"""gc_mom.py JSON — the momentum and position-edit budget of a gradient-check report, one block per control setting."""
import sys, json
r = json.load(open(sys.argv[1]))
print(f"window {r['window']}: momentum budget of the {r['rollout_steps']}-step rollout (mass {r['mass_total']:.0f}, dt {r['dt']:.5f})")
for k, v in r["momentum"].items():
    print(f"   {k}")
    print(f"      linear |P|/sum m|v| max {v['|P|/sum m|v| max']:.1e}, angular |L|/sum m|r||v| end {v['|L|/sum m|r||v| end']:.1e}, "
          f"sum m|v| max/M {v.get('sum m|v| max / M (wu/s)', float('nan')):.2e} wu/s")
    print(f"      COM moved {v['COM displacement (wu)']:.2e} wu, of which by momentum {v['COM displacement explained by momentum (wu)']:.2e}")
    if "layer |edit| median (wu)" in v:
        print(f"      outer layer ({100 * v['layer share']:.1f} % of particles): moved by velocity med {v['layer |v-displacement| median (wu)']:.2e} "
              f"p95 {v['layer |v-displacement| p95 (wu)']:.2e} wu; moved by position edits med {v['layer |edit| median (wu)']:.2e} "
              f"p95 {v['layer |edit| p95 (wu)']:.2e} wu")
        print(f"      interior: moved by velocity med {v['interior |v-displacement| median (wu)']:.2e} wu; position edits p95 {v['interior |edit| p95 (wu)']:.2e} wu")
