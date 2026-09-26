import gzip
import hashlib
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('storage_guard', Path(__file__).parents[1] / 'scripts/ops/storage_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_archive_preserves_exact_bytes_before_removal(tmp_path):
    output = tmp_path / 'output'; output.mkdir()
    src = output / 'failed.npz'
    payload = b'failed experiment evidence\n' * 10000
    src.write_bytes(payload)
    record = guard.archive_verified(src, tmp_path)
    assert not src.exists()
    assert gzip.open(record['archive'], 'rb').read() == payload
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()


def test_guard_refuses_outside_output(tmp_path):
    import pytest
    src = tmp_path / 'protected.npz'; src.write_bytes(b'x' * 1000)
    with pytest.raises(ValueError):
        guard.archive_verified(src, tmp_path)
    assert src.exists()
