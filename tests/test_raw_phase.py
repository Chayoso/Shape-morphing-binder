"""Raw-phase clock remains exact when an unphysical null-held row is present."""
import importlib.util
from pathlib import Path

import warp as wp


def test_phase_windows_keep_prior_accepted_endpoint_and_skip_null_hold(tmp_path, monkeypatch):
    previous = wp.config.kernel_cache_dir
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        'raw_phase', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        wp.config.kernel_cache_dir = previous
    records = [dict(frame_end=3), dict(frame_end=6), dict(frame_end=8)]
    assert module.phase_frame_indices(records, 1, 3, 2) == [[2, 4, 5], [5, 6, 7]]
