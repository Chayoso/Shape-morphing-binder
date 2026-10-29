"""r4_tl_read.py -- D145 reading 4: the timeline of r4_timeline.sh (display_tl.jsonl + topo_tl.jsonl) as one line per state:
components, internal cavities and their volume (the 50 components with the most particles), the main surface's
volume, particles lining a cavity, apart, particles in a separate solid piece, genus of the largest, watertight,
self-intersecting pairs. CPU; prints only."""
import json
X = "/data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/r4"
topo = {}
for s in open(X + "/topo_tl.jsonl"):
    r = json.loads(s)
    topo[r["tag"]] = r
for s in open(X + "/display_tl.jsonl"):
    r = json.loads(s)
    t = topo.get("%s_%s_%s" % (r["mesh"], r["label"], r["raw"]), {})
    cv = sum(-c["volume"] for c in r["other_components"] if c["kind"] == "cavity")
    print(r["mesh"], r["label"], r["raw"], "t=%.2f" % (r["raw"] / 240), "comp", r["components"], "cav", r["cavities"],
          "cavvol %.4f" % cv, "main %.2f" % r["main_volume"], "at_cav", r["particles_at_cavities"], "apart", r["apart"],
          "inpieces", r["particles_in_pieces"], "genus", t.get("genus_largest"), t.get("watertight"),
          t.get("self_intersecting_pairs"))
