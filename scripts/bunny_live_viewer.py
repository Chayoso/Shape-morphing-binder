"""Render accepted server checkpoints into a local SSH-tunneled experiment monitor."""
import argparse
import io
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys
import time
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph.render.material_surface import MaterialSurfaceViews
from physmorph.viewer.live_pair import LivePairMonitor


def verify_render_provenance(meta, root, native_extension):
    dependencies = ('physmorph/render/material_surface.py', 'physmorph/pipeline/gauss_loss.py',
                    'physmorph/render/covariance.py')
    for dependency in dependencies:
        actual = hashlib.sha256((Path(root)/dependency).read_bytes()).hexdigest()
        if actual != meta['source_sha256'][dependency]:
            raise ValueError(f'viewer rendering code differs from experiment: {dependency}')
    if hashlib.sha256(Path(native_extension).read_bytes()).hexdigest() != meta['raster_sha256']:
        raise ValueError('viewer native raster differs from experiment')


def read_checkpoint(run, source_surface):
    """Legacy non-atomic writers may yield incomplete ZIPs; caller retries safely."""
    path = run/'latest_state.npz'
    report_path = run/'report.json'
    if path.exists():
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as a:
            surface = a['surface'].copy()
            attempt = int(a['iteration_attempt']) if 'iteration_attempt' in a.files else None
    elif report_path.exists():
        # A zero-iteration run or failed response probe may never call on_iteration.
        # The final immutable replay is authoritative, including for older writers.
        path = run/'trajectory.npz'
        with np.load(path, allow_pickle=False) as a:
            surface, attempt = a['surface_frames'][-1].copy(), None
    else:
        return source_surface, '초기 상태 · 수락된 업데이트 대기'
    if surface.shape != source_surface.shape or not np.isfinite(surface).all():
        raise ValueError('invalid accepted surface checkpoint')
    stamp = time.strftime('%H:%M:%S UTC', time.gmtime(path.stat().st_mtime))
    status = '수락된 물리 상태 · 저장 '+stamp
    if attempt is not None:
        status += f' · 마지막 시도 {attempt+1}'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        q = report['validation']
        status += ('\n실행 종료 · 개발 검증 E128='+f"{q['point_silhouette_error_128']:.6f}"
                   +'\n미공개 시험/최종 품질 통과를 뜻하지 않음')
    return surface, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physics', required=True)
    ap.add_argument('--guided', required=True)
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--port', type=int, default=8776)
    args = ap.parse_args()
    if socket.gethostname() != 'hyde06':
        raise SystemExit('native rendering runs on hyde06; connect through an SSH tunnel')
    torch.set_num_threads(2)
    runs = [Path(args.physics), Path(args.guided)]
    metas = [json.loads((p/'metadata.json').read_text()) for p in runs]
    for key in ('fixture', 'camera_radius', 'steps', 'basis_sha256', 'raster_sha256', 'source_sha256', 'script_sha256'):
        if metas[0][key] != metas[1][key]:
            raise ValueError(f'cannot display unmatched pair: {key}')
    if [m['optimizer']['objective'] for m in metas] != ['physics', 'image']:
        raise ValueError('expected physics and image arms in that order')
    if any(m['observation'] != 'material_surface' for m in metas):
        raise ValueError('this monitor requires the material-surface observation')
    fixture_path = Path(args.fixture)/'fixture.npz'
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != metas[0]['fixture']['fixture_sha256']:
        raise ValueError('viewer fixture does not match the pair')
    with np.load(fixture_path, allow_pickle=False) as a:
        initial, faces = a['surface0'], a['surface_faces']
        target, target_faces = a['target_surface_vertices'], a['target_surface_faces']
    from physmorph.pipeline.gauss_loss import _gs
    extension = importlib.import_module(_gs()[0].__name__+'._C')
    verify_render_provenance(metas[0], Path(__file__).resolve().parents[1], extension.__file__)
    view = MaterialSurfaceViews([(np.pi/4, -.1)], metas[0]['camera_radius'], faces, target, target_faces, 1024)
    reference = view.targets[0].permute(1, 2, 0).cpu().numpy()
    monitor = LivePairMonitor(args.port)
    previous, images, statuses = [None, None], [None, None], ['', '']
    prm = metas[0]['fixture']['discretization']
    setup = (f"N={metas[0]['fixture']['N']} · dx={prm['dx']} · dt={prm['dt']:.8f}\n"
             f"{metas[0]['steps']} steps · native 1024×1024\n"
             '표면 정점은 MPM 격자 속도로 이동\n각 패널의 최적화 진행 시점은 다를 수 있음')
    try:
        while True:
            changed = False
            for i, run in enumerate(runs):
                watched = [run/'latest_state.npz', run/'report.json', run/'trajectory.npz']
                signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in watched)
                if signature == previous[i]:
                    continue
                try:
                    surface, status = read_checkpoint(run, initial)
                    with torch.no_grad():
                        image = view.render(torch.tensor(surface, device='cuda'), view.cams[0])
                    images[i], statuses[i] = image.permute(1, 2, 0).cpu().numpy(), status
                    previous[i], changed = signature, True
                except (OSError, ValueError, EOFError, zipfile.BadZipFile, json.JSONDecodeError):
                    continue
            if changed and all(x is not None for x in images):
                monitor.publish(*images, reference, {'physics': statuses[0], 'guided': statuses[1], 'setup': setup})
                print(json.dumps({'physics': statuses[0], 'guided': statuses[1]}, ensure_ascii=False), flush=True)
            time.sleep(2)
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
