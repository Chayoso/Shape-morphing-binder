"""Post-run validation for the original P303 reference-swap diagnostic.

This does not rewrite the result or claim that later guards ran in that job.
Large array checks run on CUDA; JSON/hash/archive handling is file I/O.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def finite_json(value):
    if isinstance(value,float):
        require(math.isfinite(value), 'Nonfinite report scalar')
    elif isinstance(value,dict):
        for item in value.values(): finite_json(item)
    elif isinstance(value,list):
        for item in value: finite_json(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--snapshot',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args = p.parse_args()
    require(not args.out.exists(), 'Verification output already exists')
    for path in (args.run,args.snapshot,args.out):
        require(path.resolve().is_relative_to('/data/relcfd/chayo/physmorph_v2'), 'Path outside project data')
    report_path, protocol_path = args.run/'result.json',args.run/'protocol.json'
    report = json.loads(report_path.read_bytes()); finite_json(report)
    protocol = json.loads(protocol_path.read_bytes())
    require(sha(protocol_path)==report['protocol_sha256'], 'Protocol hash mismatch')
    require(protocol['attempts']==[19,20] and report['observed_outer_acceptance'] is True
            and report['evaluator_expiration_checked'] is True, 'Missing acceptance/lifetime gates')
    files = {str(report_path):sha(report_path),str(protocol_path):sha(protocol_path)}
    files.update(protocol['inputs'])
    files.update({str(args.snapshot/key):value for key,value in protocol['code'].items()})
    files[str(args.snapshot/'scripts/probes/reference_swap.py')] = protocol['probe_sha256']
    files.update({str(args.run/key):value for key,value in report['sidecars'].items()})
    require(all(sha(Path(key))==value for key,value in files.items()), 'Evidence/source mismatch')
    rows = [r for r in report['history'] if r.get('animation')==19 and r.get('frame_end')]
    require(len(rows)==1, 'Audited outer accepted record missing')
    row = rows[0]
    replay = row['replay_diagnostics']
    require(replay['commit_source'] in ('accepted_buffer','replay'), 'Unknown commit source')
    expected = row['loss'] if replay['commit_source']=='accepted_buffer' else replay['replay_E_final']
    a = report['analysis']
    observed = a['common']['end']['value']+a['samples']['end']['new']['data']['value']
    tolerance = max(float(replay.get('replay_E_tol') or 0.),2e-7+2e-5*abs(expected))
    require(abs(observed-expected)<=tolerance, 'Final merit mismatch')
    require(a['lambda_render']==row['lambda'], 'Current lambda mismatch')
    require(a['old_kind']==a['new_kind']=='paced', 'Unexpected PBR preparation branch')
    require(a['pbr_grid']['old']==a['pbr_grid']['new'], 'PBR grid parameters differ')
    # Both packets were prepared in the paced branch of the bound source,
    # using one config/TargetPack; views/res/extent/k/ambient/w_pbr are unchanged.
    cfg = protocol['config']
    require(cfg['render_paced'] and cfg['w_pbr']==1. and not cfg.get('render_paced_onset'),
            'Cannot infer fixed PBR observation settings from this protocol')
    owned = []
    for attempt in (19,20):
        with np.load(args.run/f'attempt_{attempt}.npz',allow_pickle=False) as data:
            packet = {key:torch.as_tensor(data[key],device='cuda') for key in data.files}
        require(all(bool(torch.isfinite(v).all()) for v in packet.values()), 'Nonfinite observed array')
        require(packet['x0'].shape==(report['N'],3) and packet['positions'].shape==(report['T'],report['N'],3),
                'Invalid observed position layout')
        require(torch.equal(packet['positions'][-1],packet['xT']), 'Raw endpoint mismatch')
        owned.append(packet)
    require(torch.equal(owned[0]['xT'],owned[1]['x0']), 'Nonconsecutive endpoint states')
    require(torch.equal(owned[0]['pbr_grid_min'],owned[1]['pbr_grid_min']), 'PBR origin changed')
    gradient_shapes = {}
    with np.load(args.run/'common_and_data_gradients.npz',allow_pickle=False) as data:
        for key in data.files:
            value = torch.as_tensor(data[key],device='cuda')
            require(value.shape==(report['N'],3) and bool(torch.isfinite(value).all()), 'Invalid gradient sidecar')
            gradient_shapes[key] = list(value.shape)
    require(all(sha(Path(key))==value for key,value in files.items()), 'Evidence changed during verification')
    result = dict(kind='post-run verification; later guards were NOT executed inside the original job',
        files=files, verifier_sha256=sha(Path(__file__)), finite_scalars_and_arrays=True,
        closure=dict(commit_source=replay['commit_source'],expected=expected,reconstructed=observed,
                     absolute_error=abs(observed-expected),absolute_tolerance=tolerance),
        pbr=dict(paced_branch_same_config=True,grid_origins_equal=True,grid=a['pbr_grid'],
            config={key:cfg[key] for key in ('render_views','render_elevs','render_res','sil_k','pbr_ambient','w_pbr')}),
        gradient_shapes=gradient_shapes, N=report['N'],T=report['T'],mpm=report['mpm'],loss_res=report['loss_res'])
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result['closure']),flush=True)


if __name__=='__main__':
    main()
