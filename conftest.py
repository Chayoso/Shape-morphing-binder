"""Pytest root config: ensure repo root on path, shared fixtures."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


@pytest.fixture
def device():
    return DEVICE
