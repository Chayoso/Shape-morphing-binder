"""d154_jobs.py OUT_DIR WORKERS -- D154 (1)'s jobs for d154_mesh_read.py: per mesh-seed of B1 on the tag (d145t_b1) the
end frames of L and P, the run's own target sample and D90's independent sample, and for the runs whose kept frames
are left (bunny and dragon L97 / P97) the kept frames at t = 1, 2, 3, 4, 5, 6, 8, 10 s. Whole mesh-seeds go to one
worker (its reference is built once), the heaviest first, round robin. Writes OUT_DIR/jobs_end_K.txt."""
import sys
from pathlib import Path

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
B1 = f"{G}/d145t_b1"
out, workers = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seeds = {"bunny": (97, 99, 101, 103, 105), "dragon": (97, 99, 101, 103, 105), "teapot": (97, 99, 101), "bob": (97, 99, 101),
         "armadilo": (97, 99, 101), "homer": (97, 99, 101)}
groups = []
for mesh, ss in seeds.items():
    for s in ss:
        g = [f"{mesh} {s} L {B1}/{mesh}300k_L{s}_end1.npz end", f"{mesh} {s} P {B1}/{mesh}300k_P{s}_end1.npz end",
             f"{mesh} {s} own {B1}/{mesh}300k_L{s}_end1.npz tgt", f"{mesh} {s} ind none ind"]
        if mesh in ("bunny", "dragon") and s == 97:
            for arm in "LP":
                g += [f"{mesh} {s} {arm}@{t} {B1}/{mesh}300k_{arm}{s}_frames12.npz {240 * t}" for t in (1, 2, 3, 4, 5, 6, 8, 10)]
        groups.append(g)
groups.sort(key=len, reverse=True)
files = [[] for _ in range(workers)]
for i, g in enumerate(groups):
    min(files, key=len).extend(g)
for k, f in enumerate(files):
    (out / f"jobs_end_{k}.txt").write_text("\n".join(f) + "\n")
    print(k, len(f))
