#!/bin/bash
# B12 (D145): the render-only baselines' env (Large Steps + FlexiCubes on nvdiffrast), everything under /data.
# Own conda env by prefix; ~/miniforge3 is only executed (its conda binary), never modified: packages cached under
# $B12/cache, the env not registered in ~/.conda. Name lookups for the installs go through dnsfix/sitecustomize.py
# (hyde06's tailscale resolver fails; the fallback asks 8.8.8.8), only on this script's PYTHONPATH.
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
export CONDA_PKGS_DIRS=$B12/cache/conda_pkgs PIP_CACHE_DIR=$B12/cache/pip TMPDIR=$B12/cache/tmp XDG_CACHE_HOME=$B12/cache/xdg
export CONDA_REGISTER_ENVS=false CONDA_ENVS_PATH=$B12 TORCH_EXTENSIONS_DIR=$B12/cache/torch_ext
export PYTHONPATH=$B12/dnsfix PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 CC=gcc CXX=g++
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=16
LOG=$B12/logs/env.log
{
echo "start $(date '+%F %T')"
python3 -c "import socket" 2>/dev/null
/home/chayo/miniforge3/bin/conda create -y --prefix $B12/env -c conda-forge --override-channels python=3.10 pip || exit 1
P=$B12/env/bin/python
$P -c "import sitecustomize, socket; print('dnsfix', socket.getaddrinfo('pypi.org', 443)[0][4])" || exit 2
$P -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || exit 3
$P -m pip install numpy==1.26.4 scipy==1.15.3 trimesh==4.11.5 largesteps==0.2.2 cholespy gpytoolbox libigl==2.6.3 \
    imageio tqdm ninja setuptools wheel || exit 4
cd $B12/src/nvdiffrast-HEAD && $P -m pip install . --no-build-isolation || exit 5
$P -m pip freeze > $B12/logs/env_freeze.txt
cd $B12
PYTHONPATH= $P -c "import torch, nvdiffrast, nvdiffrast.torch as dr, largesteps, cholespy, gpytoolbox, igl, trimesh, numpy; print('torch', torch.__version__, torch.version.cuda, 'nvdiffrast', nvdiffrast.__version__, 'gpytoolbox', gpytoolbox.__version__ if hasattr(gpytoolbox, '__version__') else '?', 'numpy', numpy.__version__)" || exit 6
echo "end $(date '+%F %T')"
} > $LOG 2>&1
echo "exit $?" >> $LOG
