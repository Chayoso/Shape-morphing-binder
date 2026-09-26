import numpy as np
import pytest
import torch
from physmorph.render.settled import SettledAppearance, pin_start_frames, validate_pinned_frames


def test_attributes_lock_independently_at_admission_and_motion_is_not_hidden():
    starts = pin_start_frames(np.array([True, True]), np.array([1, 2]),
                             [dict(animation=0, frame_end=3), dict(animation=1, frame_end=5)], {})
    state = SettledAppearance(starts, 'cpu')
    x = torch.zeros(2, 3)
    def frame(i):
        return state.apply(i, x, torch.full((2, 3), float(i)), torch.full((2,), float(i)), torch.full((2,), float(i)))
    assert frame(1)[1].tolist() == [1, 1]
    assert frame(2)[1].tolist() == [2, 2]
    assert frame(3)[1].tolist() == [2, 3]
    assert frame(4)[1].tolist() == [2, 4]
    assert frame(5)[1].tolist() == [2, 4]
    assert frame(5)[2].tolist() == [5, 5]  # current density must never be hidden by the latch
    x[0, 0] = 0.01
    with pytest.raises(ValueError, match='active pin moved'):
        frame(6)


def test_released_pins_require_history_and_cannot_silently_lock():
    with pytest.raises(ValueError, match='release modes'):
        pin_start_frames(np.array([True]), np.array([1]), [], dict(settle_pin_yield=True))


def test_raw_validation_catches_motion_before_first_rendered_settled_frame():
    frames = np.zeros((5, 2, 3), np.float32)
    frames[2:, 0, 0] = 0.1  # motion between admission at1 and sampled frame4
    with pytest.raises(ValueError, match='raw frame 2'):
        validate_pinned_frames(frames, np.array([1, 100]), 5)
