import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts.probes.normalize_disabled_pins import normalize


def make_run(root, **config):
    prefix = root/'source'
    arm = dict(config=dict(settle_pin=False, **config), history=[dict(pinned_frac=0.)], guards={})
    meta = dict(**arm, arms={'render_full_dt_iso_nn':arm})
    prefix.with_suffix('.json').write_text(json.dumps(meta))
    prefix.with_suffix('.log').write_text('kept')
    np.savez(prefix.with_suffix('.npz'), final=np.zeros((4, 3)))
    np.savez(str(prefix)+'_render_full_dt_iso_nn.npz', src=np.zeros((4, 3)),
             frames=np.ones((2, 4, 3)), pinned=None, pinned_at=None)
    return prefix


def test_disabled_pin_schema_keeps_numeric_payloads_and_metadata_exact(tmp_path):
    source = make_run(tmp_path)
    target = tmp_path/'normalized'
    normalize(source, target)
    suffix = '_render_full_dt_iso_nn.npz'
    with zipfile.ZipFile(str(source)+suffix) as a, zipfile.ZipFile(str(target)+suffix) as b:
        assert set(a.namelist()) == set(b.namelist())
        for name in ('src.npy', 'frames.npy'):
            assert a.read(name) == b.read(name)
    assert source.with_suffix('.json').read_bytes() == target.with_suffix('.json').read_bytes()
    with np.load(str(target)+suffix, allow_pickle=False) as data:
        assert data['pinned'].dtype == np.bool_ and not data['pinned'].any()
        assert data['pinned_at'].dtype == np.int64 and (data['pinned_at'] == -1).all()
    with pytest.raises(FileExistsError):
        normalize(source, target)


def test_disabled_pin_schema_rejects_another_freezing_policy(tmp_path):
    source = make_run(tmp_path, freeze_arrived=True)
    with pytest.raises(ValueError):
        normalize(source, tmp_path/'normalized')
    assert not (tmp_path/'normalized_render_full_dt_iso_nn.npz').exists()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.1])
def test_disabled_pin_schema_rejects_invalid_fraction(tmp_path, value):
    source = make_run(tmp_path)
    meta = json.loads(source.with_suffix('.json').read_text())
    meta['history'][0]['pinned_frac'] = value
    meta['arms']['render_full_dt_iso_nn']['history'][0]['pinned_frac'] = value
    source.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        normalize(source, tmp_path/'normalized')
