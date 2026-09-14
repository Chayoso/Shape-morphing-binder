import base64
import json
import urllib.request
import urllib.error

import pytest
from physmorph.viewer.paired import PairedReplay


def setup_pair(tmp_path):
    folders = [tmp_path/'physics', tmp_path/'guided']
    for n, p in enumerate(folders):
        p.mkdir()
        (p/'target.png').write_bytes(b'same target')
        (p/'0.png').write_bytes(bytes([n, 0]))
        (p/'1.png').write_bytes(bytes([n, 1]))
        (p/'metadata.json').write_text(json.dumps({'discretization':{'dt':.01}}))
        (p/'qa_manifest.json').write_text(json.dumps({'frame_count':2,'files':['0.png','1.png'],'visual_review':'reviewed all frames'}))
    return folders


def test_paired_snapshot_keeps_the_same_physical_time(tmp_path):
    folders = setup_pair(tmp_path)
    viewer = PairedReplay(*folders, port=0)
    try:
        base = f'http://127.0.0.1:{viewer.port}'
        packet = json.load(urllib.request.urlopen(base+'/frame/1'))
        assert packet['simulation_time'] == .01
        assert base64.b64decode(packet['physics']) == bytes([0, 1])
        assert base64.b64decode(packet['guided']) == bytes([1, 1])
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base+'/frame/2')
    finally:
        viewer.close()


def test_paired_viewer_rejects_unreviewed_frames(tmp_path):
    folders = setup_pair(tmp_path)
    p = folders[0]/'qa_manifest.json'
    m = json.loads(p.read_text()); m['visual_review'] = 'pending'; p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='review'):
        PairedReplay(*folders, port=0)
