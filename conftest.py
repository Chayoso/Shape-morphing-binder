"""Pytest root config: ensure repo root on path, shared fixtures."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import physmorph  # noqa: E402,F401  (before torch: CuPy's CUDA 12 NVRTC loads first, physmorph/__init__.py)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


@pytest.fixture
def device():
    return DEVICE
