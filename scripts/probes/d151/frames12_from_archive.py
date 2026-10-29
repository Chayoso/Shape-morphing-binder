# frames12_from_archive.py ARCHIVE OUT -- D149's kept-frames file (pipeline_run.py of repo_d149s: every 12th delivered
# frame and the last; frames, raws, tgt, deliver_n) from a full archive (NAME_<ARM>.npz of the frozen writer), the same
# arrays: the full archive's frames are np.stack(frames.x), the kept file's frames.x[k] for k in keep.
import os, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
z = np.load(src, allow_pickle=True)
dn = int(z["deliver_n"])
fr = z["frames"]
keep = sorted(set(list(range(0, dn, 12)) + [dn - 1]))
tmp = dst[:-4] + ".part.npz"
np.savez(tmp, frames=np.ascontiguousarray(fr[keep]), raws=np.asarray(keep), tgt=z["tgt"], deliver_n=np.int64(dn))
os.replace(tmp, dst)
print(f"{dst}: {len(keep)} of {fr.shape[0]} frames (deliver_n {dn}), {fr.dtype}")
