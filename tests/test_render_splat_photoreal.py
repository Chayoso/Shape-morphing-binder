"""Guard against concealing raw motion when exporting temporally subsampled video."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        'render_splat_photoreal', Path(__file__).resolve().parents[1] / 'scripts' / 'render_splat_photoreal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unrendered_transient_pin_motion_refuses_export():
    renderer = load_renderer()
    frames = np.zeros((4, 2, 3), np.float32)
    # A renderer seeing only frames 0 and 3 would incorrectly certify immobility.
    frames[2, 0, 1] = .01
    with pytest.raises(ValueError, match='raw frame 2'):
        renderer.validate_pins_cuda(frames, np.array([1, 999]), 4, 'cpu')


def test_unpinned_motion_and_pre_admission_motion_are_allowed():
    renderer = load_renderer()
    frames = np.zeros((4, 2, 3), np.float32)
    frames[0, 0] = 10
    frames[:, 1, 0] = np.arange(4)
    renderer.validate_pins_cuda(frames, np.array([1, 999]), 4, 'cpu')
