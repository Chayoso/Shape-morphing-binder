"""target_ref.py ARCHIVE_NPZ OUT_PREFIX — a one-frame archive whose particles are the run's own target sample, for the
4K renderer: what the renderer draws when every particle sits exactly on the target (the renderer's and the
sampling's share of the blur; the rest of a morph frame's blur is the particle arrangement)."""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
tgt = np.asarray(z["tgt"], np.float32)
out = sys.argv[2]
np.savez(out + "_render_full_dt_iso_nn.npz", src=tgt, tgt=tgt, orient=z["orient"], frames=tgt[None], deliver_n=np.int64(1))
open(out + ".json", "w").write(json.dumps({"arms": {"render_full_dt_iso_nn": {"history": [], "config": {}}}}))
print("wrote", out, tgt.shape)
