"""d157_jobs.py OUT_300K OUT_40K -- D157's job lists: per mesh-seed the own sample, the render arm and its twin.

300k: B1's end frames (d145t_b1/MESH300k_{L,P}SEED_end1.npz). 40k: the tag's gallery, d154/g40runs/MESH40k_{V,W}SEED_end1.npz
(14 meshes) and D155's gate re-runs' kept frames (d155/gate/MESH40k_{V,W}SEED_frames12.npz, the last one) for the five
meshes D154 (3) read there. Lines: MESH SEED N LABEL NPZ RAW."""
import os
import sys

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
B1 = {"bunny": (97, 99, 101, 103, 105), "dragon": (97, 99, 101, 103, 105), "teapot": (97, 99, 101),
      "bob": (97, 99, 101), "armadilo": (97, 99, 101), "homer": (97, 99, 101)}
GATE = ("armadilo", "bunny", "dragon", "homer", "teapot")
G40 = ("A", "beast", "bimba", "bob", "C", "cheburashka", "cow", "fandisk", "heart", "maxplanck", "nefertiti", "ogre",
       "spot", "V")

with open(sys.argv[1], "w") as f:
    for mesh, seeds in B1.items():
        for s in seeds:
            L, P = f"{G}/d145t_b1/{mesh}300k_L{s}_end1.npz", f"{G}/d145t_b1/{mesh}300k_P{s}_end1.npz"
            assert os.path.exists(L) and os.path.exists(P), (L, P)
            f.write(f"{mesh} {s} 300k own {L} tgt\n{mesh} {s} 300k L {L} end\n{mesh} {s} 300k P {P} end\n")
with open(sys.argv[2], "w") as f:
    for mesh in sorted(GATE + G40):
        for s in (97, 99, 101):
            if mesh in GATE:
                V, W = f"{G}/d155/gate/{mesh}40k_V{s}_frames12.npz", f"{G}/d155/gate/{mesh}40k_W{s}_frames12.npz"
            else:
                V, W = f"{G}/d154/g40runs/{mesh}40k_V{s}_end1.npz", f"{G}/d154/g40runs/{mesh}40k_W{s}_end1.npz"
            assert os.path.exists(V) and os.path.exists(W), (V, W)
            f.write(f"{mesh} {s} 40k own {V} tgt\n{mesh} {s} 40k V {V} end\n{mesh} {s} 40k W {W} end\n")
