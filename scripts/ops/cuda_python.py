"""Run a CUDA pipeline script with CuPy's CUDA 12 compiler initialized first.

Some existing Torch environments contain an additional CUDA 11 NVRTC library.
Initialize the configured CUDA 12 NVRTC before importing those packages.
"""
import runpy
import sys
import cupy
from cupy_backends.cuda.libs import nvrtc

if nvrtc.getVersion()[0] < 12:
    raise RuntimeError('CUDA pipeline needs CUDA >=12 NVRTC; check CUDA_HOME/PATH and LD_LIBRARY_PATH')
if len(sys.argv) < 2:
    raise SystemExit('usage: cuda_python.py SCRIPT [ARGS...]')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
