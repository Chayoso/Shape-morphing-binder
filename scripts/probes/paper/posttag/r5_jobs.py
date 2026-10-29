"""r5_jobs.py OUT_DIR -- D145 reading 5 (2): r5_d157x.py's job files, one per mesh (OUT_DIR/jobs_MESH.txt): first our
L97 through the same code (the replication row against D157's own), then every external end frame of reading 4
(d145_ext/r4/end1, route pts), then each mesh method's own end mesh (route mesh): B14's ISD / 4Deform
(baselines/b14/recon/MESH300k_LAB/ourframe_time_20.ply), Buonomo et al. and NISE (baselines/buonomo/recon/MESH_LAB/
ourframe_time_20.ply), ARAP (baselines/arap/results/MESH_arap_t20.ply), B12's Large Steps / FlexiCubes
(baselines/b12/results/{ls,fc}/MESH_{a,b}_sSEED/frames.npz). Missing files are listed, not guessed."""
import glob
import os
import re
import sys

B = "/data/relcfd/chayo/physmorph_v2"
G = f"{B}/output/gpu"
A = f"{B}/baselines"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for mesh in ("bunny", "dragon", "teapot", "bob", "armadilo", "homer"):
    lines = [f"{mesh} pts L {G}/d145t_b1/{mesh}300k_L97_end1.npz"]
    labs = sorted(os.path.basename(f)[len(mesh) + 5:-len("_end1.npz")] for f in glob.glob(f"{G}/d145_ext/r4/end1/{mesh}300k_*_end1.npz"))
    lines += [f"{mesh} pts {lab} {G}/d145_ext/r4/end1/{mesh}300k_{lab}_end1.npz" for lab in labs]
    missing = []
    for lab in labs:
        if re.match(r"(ISD|FD)", lab):
            p = f"{A}/b14/recon/{mesh}300k_{lab}/ourframe_time_20.ply"
        elif re.match(r"(BUO|NISE)", lab):
            p = f"{A}/buonomo/recon/{mesh}_{lab}/ourframe_time_20.ply"
        elif lab == "ARAP":
            p = f"{A}/arap/results/{mesh}_arap_t20.ply"
        elif re.match(r"(LS|FC)[ab]\d+$", lab):
            m = re.match(r"(LS|FC)([ab])(\d+)$", lab)
            p = f"{A}/b12/results/{m.group(1).lower()}/{mesh}_{m.group(2)}_s{m.group(3)}/frames.npz"
        else:
            continue                                      # particles only (Xu et al., OT)
        if os.path.exists(p):
            lines.append(f"{mesh} mesh {lab} {p}")
        else:
            missing.append(p)
    open(f"{out}/jobs_{mesh}.txt", "w").write("\n".join(lines) + "\n")
    print(mesh, len(lines), "jobs;", "missing:", missing)
