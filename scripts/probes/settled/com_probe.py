"""com_probe.py NPZ [NPZ ...] — linear momentum over a whole delivered morph, from positions only.
Uniform particle mass and no external force (f_ext = 0, the domain far from the body), and the morph starts at rest:
if total momentum were conserved the centre of mass (COM) would not move at all. Reported per run: the source and
target COMs and their offset (what a morph has to move the COM by, if the target is not re-centred), the COM path
(displacement at 10 %, 25 %, 50 %, 100 % of the frames), the largest single-frame COM step, and the COM velocity
implied between the first and last 5 % of the frames, all in world units and in target spacings. Frame-to-frame COM
velocity x frame rate would be the momentum per unit mass; with positions only the frame rate cancels in ratios."""
import sys, numpy as np
from scipy.spatial import cKDTree
from physmorph.sampling.orientation import orient_archive
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    frames, tgt, src, orient = orient_archive(z, path)
    tgt = np.asarray(tgt, np.float32)
    dn = min(len(frames), int(z["deliver_n"])) if "deliver_n" in z.files else len(frames)
    sp = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    com = np.array([np.asarray(frames[i], np.float64).mean(0) for i in range(dn)])
    c_src = np.asarray(src, np.float64).mean(0) if src is not None else com[0]
    c_tgt = tgt.astype(np.float64).mean(0)
    d = np.linalg.norm(com - com[0], axis=1)
    step = np.linalg.norm(np.diff(com, axis=0), axis=1)
    q = lambda f: d[min(dn - 1, int(round(f * (dn - 1))))]
    print(f"{path.split('/')[-1]}: {dn} frames, target spacing {sp:.4f} wu")
    print(f"   COM source {np.round(c_src, 4)}, frame 0 {np.round(com[0], 4)}, target {np.round(c_tgt, 4)}; "
          f"target - start offset {np.linalg.norm(c_tgt - com[0]):.4f} wu ({np.linalg.norm(c_tgt - com[0]) / sp:.1f} sp)")
    print(f"   COM displacement from frame 0 at 10/25/50/100 %: {q(.1):.4f} {q(.25):.4f} {q(.5):.4f} {q(1.):.4f} wu "
          f"(end = {q(1.) / sp:.2f} sp); end COM - target COM {np.linalg.norm(com[-1] - c_tgt):.4f} wu; "
          f"max single-frame COM step {step.max():.2e} wu at frame {int(step.argmax()) + 1}")
    print(f"   end COM direction {np.round((com[-1] - com[0]) / max(np.linalg.norm(com[-1] - com[0]), 1e-12), 3)}, "
          f"target-offset direction {np.round((c_tgt - com[0]) / max(np.linalg.norm(c_tgt - com[0]), 1e-12), 3)}")
