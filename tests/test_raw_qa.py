import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('raw_qa', Path(__file__).parents[1] / 'scripts/probes/morph_raw_qa.py')
qa = importlib.util.module_from_spec(spec); spec.loader.exec_module(qa)


@pytest.mark.parametrize('compressed', [False, True])
def test_npz_frames_are_read_exactly(tmp_path, compressed):
    frames = np.random.default_rng(3).normal(size=(3, 100, 3)).astype(np.float32)
    path = tmp_path / 'frames.npz'
    saver = np.savez_compressed if compressed else np.savez
    saver(path, src=frames[0], frames=frames)
    assert np.array_equal(qa.frames_array(path), frames)


def test_pin_audit_detects_motion_after_admission(tmp_path):
    prefix = tmp_path / 'trial'
    x = np.random.default_rng(5).normal(size=(100, 3)).astype(np.float32)
    frames = np.stack([x, x, x]); frames[2, 4, 0] += 0.1
    pins = np.zeros(100, bool); pins[4] = True
    at = np.full(100, -1); at[4] = 1
    np.savez(str(prefix)+'_'+qa.ARM+'.npz', src=x, tgt=x, frames=frames, pinned=pins,
             pinned_at=at, deliver_n=3)
    Path(str(prefix)+'.json').write_text(json.dumps({'arms': {qa.ARM: dict(
        config=dict(T=1, animations=2), history=[dict(animation=0, frame_end=2),
        dict(animation=1, frame_end=3)], metrics={}, guards={})}}))
    result = qa.audit(prefix)
    assert result['pin_motion']['checked_particles'] == 1
    assert result['pin_motion']['max_wu'] == pytest.approx(0.1, abs=1e-6)


def test_pins_admitted_after_delivery_are_not_counted_as_active(tmp_path):
    prefix = tmp_path / 'truncated'
    x = np.random.default_rng(9).normal(size=(100, 3)).astype(np.float32)
    frames = np.stack([x]*6)
    np.savez(str(prefix)+'_'+qa.ARM+'.npz', src=x, tgt=x, frames=frames,
             pinned=np.ones(100, bool), pinned_at=np.full(100, 2), deliver_n=4)
    Path(str(prefix)+'.json').write_text(json.dumps({'arms': {qa.ARM: dict(
        config=dict(T=1, animations=2), history=[dict(animation=0, frame_end=3),
        dict(animation=1, frame_end=6)], metrics={}, guards={})}}))
    result = qa.audit(prefix)
    assert result['active_pins_at_delivered_end'] == 0
    assert result['pin_motion']['checked_particles'] == 0
