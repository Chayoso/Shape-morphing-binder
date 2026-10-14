# env_d145.sh -- sourced by every D145 script before anything else: every cache and temp dir on /data, nothing new in
# /home or /tmp (hyde06's root filesystem filled on 2026-10-14). The Python side also sources hyde06_env.sh, which sets
# the same; this covers the shell tools (sort, here-documents) and any Python started outside it.
export TMPDIR=/data/relcfd/chayo/physmorph_v2/tmp/d145/tmpdir
export XDG_CACHE_HOME=/data/relcfd/chayo/physmorph_v2/cache
export WARP_CACHE_PATH=$XDG_CACHE_HOME/warp CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda TORCH_HOME=$XDG_CACHE_HOME/torch
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib CUPY_CACHE_DIR=$XDG_CACHE_HOME/cupy PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
mkdir -p $TMPDIR
