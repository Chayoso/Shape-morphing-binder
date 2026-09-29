"""Evidence checks must fail on altered state, not only malformed metadata."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.probes.window_selection import IdentityCapture, exact_tree


def test_exact_tree_rejects_one_ulp_and_missing_keys():
    reference = ({'X': np.array([1.], np.float32), 'F': torch.eye(3)}, [None, 2.])
    assert exact_tree(deepcopy(reference), reference)
    changed = deepcopy(reference)
    changed[0]['X'][0] = np.nextafter(np.float32(1), np.float32(2))
    assert not exact_tree(changed, reference)
    changed = deepcopy(reference)
    del changed[0]['F']
    assert not exact_tree(changed, reference)


@pytest.mark.parametrize('corrupt', [False, True])
def test_handoff_checks_new_pin_zeroing_and_surviving_free_state(tmp_path, corrupt):
    capture = IdentityCapture(tmp_path)
    I = torch.eye(3).repeat(3, 1, 1)
    capture.start = dict(Fp=I.clone(), pin=torch.tensor([1., 0., 0.]))
    capture.head = dict(x=torch.arange(9).reshape(3, 3).float(), F=I.clone(),
                        v=torch.ones(3, 3), C=torch.ones(3, 3, 3))
    capture.successor = {key+'0': value.clone() for key, value in capture.head.items()}
    capture.successor.update(Fp=I*1.01, pin=torch.tensor([1., 1., 0.]))
    capture.successor['v0'][:2] = 0.
    capture.successor['C0'][:2] = 0.
    if corrupt:
        capture.successor['v0'][2, 0] = 0.
        with pytest.raises(ValueError, match='Whole-state handoff mismatch'):
            capture.check_handoff()
    else:
        capture.check_handoff()
        assert capture.receipt['handoff']['new_pins'] == 1
        assert capture.receipt['handoff']['surviving_free'] == 1
        assert capture.start is capture.head is capture.successor is None


def test_encoded_report_budget_rejects_before_writing_either_file(tmp_path, monkeypatch):
    from physmorph.pipeline.render_reporting import write_render_report
    from scripts.probes import window_selection
    monkeypatch.setattr(window_selection, 'LIMIT', 100)
    capture = IdentityCapture(tmp_path)
    capture.write_json('existing.json', {'small': True})
    before = list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='reserved output bytes'):
        capture.write_json('too_large.json', {'text': 'x'*100})
    with pytest.raises(ValueError, match='reserved output bytes'):
        write_render_report(tmp_path/'report', [], {}, {}, 3, reserve_bytes=capture.reserve)
    assert list(tmp_path.iterdir()) == before
