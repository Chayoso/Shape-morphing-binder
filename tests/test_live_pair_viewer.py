import base64
import io
import json
import hashlib
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image

from physmorph.viewer.live_pair import LivePairMonitor
from scripts.bunny_live_viewer import read_checkpoint, verify_render_provenance


def test_live_pair_publishes_consistent_native_images_and_revisions():
    viewer = LivePairMonitor(port=0)
    try:
        images = [np.full((12, 12, 3), v) for v in (.1, .5, .9)]
        for revision in (1, 2):
            viewer.publish(*images, {'physics': str(revision), 'guided': 'pending', 'setup': 'CPU fixture'})
            with urlopen(f'http://127.0.0.1:{viewer.port}/snapshot') as response:
                packet = json.load(response)
            assert packet['version'] == revision and packet['status']['physics'] == str(revision)
            for key, expected in zip(('physics', 'guided', 'target'), images):
                actual = np.asarray(Image.open(io.BytesIO(base64.b64decode(packet[key]))))
                assert np.array_equal(actual, (expected*255).round().astype(np.uint8))
    finally:
        viewer.close()


def test_completed_run_without_iteration_callback_uses_final_replay(tmp_path):
    initial = np.zeros((4, 3), np.float32)
    final = np.ones_like(initial)
    (tmp_path/'report.json').write_text(json.dumps({'validation': {'point_silhouette_error_128': .2}}))
    np.savez_compressed(tmp_path/'trajectory.npz', surface_frames=np.stack([initial, final]))
    surface, status = read_checkpoint(tmp_path, initial)
    assert np.array_equal(surface, final)
    assert '0.200000' in status and '실행 종료' in status


def test_viewer_rejects_its_own_changed_render_code_or_native_binary(tmp_path):
    names = ('physmorph/render/material_surface.py', 'physmorph/pipeline/gauss_loss.py',
             'physmorph/render/covariance.py')
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    native = tmp_path/'native.bin'; native.write_bytes(b'native')
    meta = {'source_sha256': {n: hashlib.sha256(b'original').hexdigest() for n in names},
            'raster_sha256': hashlib.sha256(b'native').hexdigest()}
    verify_render_provenance(meta, tmp_path, native)
    (tmp_path/names[0]).write_bytes(b'changed covariance')
    with pytest.raises(ValueError, match='rendering code'):
        verify_render_provenance(meta, tmp_path, native)
    (tmp_path/names[0]).write_bytes(b'original')
    native.write_bytes(b'different raster')
    with pytest.raises(ValueError, match='native raster'):
        verify_render_provenance(meta, tmp_path, native)
