"""d156_jobs.py OUT_JOBS [MESH ...] -- D156: the reader's job list for the 40k states (d156_read.py's format).

Per mesh and seed (97, 99), two contexts:
  hull: the D156 runs (output/gpu/d156/runs40/MESH40k_{L,P}_SEED_frames12.npz) and the perfect samples H, M, Mp;
  mesh: the tag's 40k runs with the mesh as target, V and W (D155's gate frames for armadilo, bunny, dragon, homer;
        D154 (3b)'s g40runs end frames for the others), and H, M, Mp in their frame.
The reference archives are found by glob; a mesh-seed-arm with no single match is reported and left out."""
import glob
import sys

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
MESHES = sys.argv[2:] or ["bunny", "dragon", "armadilo", "homer", "cow", "bob", "cheburashka", "C", "A"]
GATE = ("armadilo", "bunny", "dragon", "homer", "teapot")


def ref(mesh, arm, seed):
    """The tag's 40k run of MESH, arm V or W, at SEED: D155's gate frame (armadilo, bunny, dragon, homer, teapot) or
    D154 (3b)'s end frame."""
    for p in ([f"{G}/d155/gate/{mesh}40k_{arm}{seed}_frames12.npz"] if mesh in GATE else []) + \
             [f"{G}/d154/g40runs/{mesh}40k_{arm}{seed}_end1.npz"]:
        if glob.glob(p):
            return p
    print(f"missing {mesh} {arm} {seed}", file=sys.stderr)
    return None


lines = []
for seed in (97, 99):
    for mesh in MESHES:
        L = f"{G}/d156/runs40/{mesh}40k_L_{seed}_frames12.npz"
        P = f"{G}/d156/runs40/{mesh}40k_P_{seed}_frames12.npz"
        lines += [f"{mesh} {seed} hull {lab} {L} ref" for lab in ("H", "M", "Mp")]
        lines += [f"{mesh} {seed} hull L {L} end", f"{mesh} {seed} hull P {P} end"]
        V, W = ref(mesh, "V", seed), ref(mesh, "W", seed)
        if V and W:
            lines += [f"{mesh} {seed} mesh {lab} {V} ref" for lab in ("H", "M", "Mp")]
            lines += [f"{mesh} {seed} mesh V {V} end", f"{mesh} {seed} mesh W {W} end"]
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} jobs -> {sys.argv[1]}")
