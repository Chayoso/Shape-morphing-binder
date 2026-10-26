"""Small status/record compatibility tests; run only on hyde06j."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

B = Path('/data/relcfd/chayo/physmorph_v2')
reader_path = B / 'tmp/d166/d166_read.py'
spec = importlib.util.spec_from_file_location('reader_legacy_rc', reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
results = []
with tempfile.TemporaryDirectory(prefix='legacy_rc_', dir=B / 'tmp/d166/code_audit_20261023') as tmp:
    root = Path(tmp)
    prefix = root / 'bob40k_V97'
    prefix.with_suffix('.json').write_text(json.dumps({'arms': {'full': {
        'deliver_n': 41, 'history': [{'frame_end': 41, 'v_mean': .001}],
        'metrics': {'sil_iou': .99, 'chamfer': .01, 'thin_uncovered': .01}}}}))
    for legacy in (False, True):
        path = root / ('rc_bob40k_V97' if legacy else 'bob40k_V97.rc')
        cases = [('0\n', True), ('3\n', False), ('0 119\n', legacy),
                 ('7 119\n', False), ('0 -1\n', False), ('0 nan\n', False),
                 ('0 1.5\n', False), ('0 1 extra\n', False), ('\n', False),
                 ('garbage\n', False), ('0 0\n', legacy)]
        for value, expected in cases:
            path.write_text(value)
            actual = reader.run_rec(str(prefix)) is not None
            assert actual == expected, (legacy, value, actual, expected)
            results.append({'legacy': legacy, 'contents': value.strip(), 'accepted': actual})
        path.unlink()
    prefix.with_suffix('.rc').write_text('1\n')
    (root / 'rc_bob40k_V97').write_text('0 119\n')
    assert reader.run_rec(str(prefix)) is None
    results.append({'test': 'conflicting_status_files_fail', 'passed': True})
    prefix.with_suffix('.rc').write_text('0\n')
    assert reader.run_rec(str(prefix)) is not None
    results.append({'test': 'both_successful_status_files_pass', 'passed': True})
out = {'passed': len(results), 'reader_sha256': hashlib.sha256(reader_path.read_bytes()).hexdigest(), 'results': results}
(B / 'tmp/d166/code_audit_20261023/legacy_rc_results.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'passed': len(results), 'reader_sha256': out['reader_sha256']}))
