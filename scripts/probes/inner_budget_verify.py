"""Bind and verify a completed inner-budget audit without modifying its evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    for path in (args.run,args.snapshot,args.out):
        if not path.resolve().is_relative_to('/data/relcfd/chayo/physmorph_v2'):
            raise ValueError('Evidence outside project data')
    report = json.loads((args.run/'result.json').read_bytes())
    json.dumps(report,allow_nan=False)
    protocol = json.loads((args.run/'protocol.json').read_bytes())
    files = {str(args.run/'result.json'):digest(args.run/'result.json'),
             str(args.run/'protocol.json'):report['protocol_sha256'],**protocol['inputs']}
    files.update({str(args.snapshot/key):value for key,value in protocol['code'].items()})
    files[str(args.snapshot/'scripts/probes/inner_budget.py')] = protocol['probe_sha256']
    files.update({str(args.run/key):value for key,value in report['sidecars'].items()})
    if any(digest(Path(key))!=value for key,value in files.items()):
        raise ValueError('Evidence hash mismatch')
    counts = {}
    for name in report['sidecars']:
        with np.load(args.run/name,allow_pickle=False) as data:
            count = 0
            for key in data.files:
                value = torch.as_tensor(data[key],device='cuda')
                if not bool(torch.isfinite(value).all()): raise ValueError('Nonfinite sidecar: '+key)
                count += value.numel()
            count_pins = torch.as_tensor(data['pins'],device='cuda').bool()
            positions = torch.as_tensor(data['positions'],device='cuda')
            initial = torch.as_tensor(data['x0'],device='cuda')
            if not torch.equal(positions[:,count_pins],initial[count_pins][None].expand(len(positions),-1,-1)):
                raise ValueError('Start pins moved within checkpoint trajectory')
            counts[name] = dict(finite_elements=count,exact_start_pins=int(count_pins.sum()))
    if any(digest(Path(key))!=value for key,value in files.items()):
        raise ValueError('Evidence changed during verification')
    output = dict(scope='Post-run numerical array and evidence binding; does not retrofit later source guards',
                  files=files,arrays=counts,script_sha256=digest(Path(__file__)))
    with args.out.open('x') as stream: json.dump(output,stream,indent=2)
    print(json.dumps(counts),flush=True)


if __name__=='__main__': main()
