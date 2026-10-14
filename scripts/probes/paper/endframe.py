"""endframe.py FRAMES12_NPZ OUT_NPZ -- the last kept frame of a frames12 archive as its own archive (frames, raws, tgt,
deliver_n), for the end-frame readings of D145 (as d144_chain.sh's inline step)."""
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
np.savez(sys.argv[2], frames=z["frames"][-1:], raws=z["raws"][-1:], tgt=z["tgt"], deliver_n=z["deliver_n"])
