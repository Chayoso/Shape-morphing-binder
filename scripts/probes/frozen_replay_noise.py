"""Read-only C/trajectory/derivative replay diagnostic on an owned failed capture."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution,to_array
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.prepared_reference import PreparedReference
from scripts.probes.braking_capture import signature
from scripts.probes.reference_swap import require,sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    for path in (a.archive,a.out):
        require(path.resolve().is_relative_to('/data/relcfd/chayo/physmorph_v2'),'Outside project data')
    digest = sha(a.archive)
    witnesses = {}
    with cuda_execution('cuda:0'):
        for kind in ('torch_spec','cupy_spec'):
            model,packet = FrozenBodyWindow.load(a.archive,'cuda:0')
            if kind=='cupy_spec':
                def convert(value):
                    if torch.is_tensor(value): return to_array(value,copy=True)
                    if isinstance(value,(list,tuple)): return type(value)(convert(v) for v in value)
                    return value
                for f in fields(model.spec): setattr(model.spec,f.name,convert(getattr(model.spec,f.name)))
            reference = PreparedReference(**packet['reference'])
            arrived = packet['start_arrived'] & ~packet['pins']
            for label,terminal in [('baseline',model.coefficients[:,3:]),('trial05',packet['trial05_terminal'])]:
                for repeat in range(3):
                    witnesses[f'{kind}_{label}_{repeat}'] = signature(model,reference,terminal,arrived)
            model.close()
        rows = []
        eps = torch.finfo(torch.float32).eps
        for key,witness in witnesses.items():
            label = 'baseline' if '_baseline_' in key else 'trial05'
            baseline = witnesses['torch_spec_'+label+'_0']
            row = dict(witness=key,arrays={},gradients={},data={})
            for name,values in witness['arrays'].items():
                ref = baseline['arrays'][name]
                unit = dict(positions=model.spec.prm.dx,V=model.spec.prm.dx/(model.spec.T*model.spec.prm.dt),
                            F=1.,C=1/(model.spec.T*model.spec.prm.dt))[name]
                error = (values-ref).abs()
                tolerance = 32*eps*(unit+ref.abs())
                row['arrays'][name] = dict(max_abs=float(error.max()),rms=float(error.square().mean().sqrt()),
                    tolerance_ratio=float((error/tolerance).max()),failed_elements=int((error>tolerance).sum()),
                    ref_absmax=float(ref.abs().max()))
            for name,g in witness['gradients'].items():
                ref = baseline['gradients'][name]
                error = float((g-ref).norm());norm=float(ref.norm())
                row['gradients'][name] = dict(error_norm=error,reference_norm=norm,
                    tolerance_ratio=error/(64*eps*max(norm,1e-12)))
            for name,value in witness['terms'].items():
                ref = baseline['terms'][name]
                row['data'][name] = dict(value=value,reference=ref,
                    tolerance_ratio=abs(value-ref)/(32*eps*max(abs(ref),1e-12)))
            rows.append(row)
            print(json.dumps(row,allow_nan=False),flush=True)
    require(sha(a.archive)==digest,'Owned archive changed')
    root = Path(__file__).resolve().parents[2]
    code = {str(f.relative_to(root)):sha(f) for f in sorted((root/'physmorph').rglob('*.py'))}
    output = dict(archive_sha256=digest,rows=rows,code=code,script_sha256=sha(__file__),
        scope='Within-loaded repeated signatures and Torch/CuPy initial-array representation; original failed in-memory signature unavailable')
    with a.out.open('x') as stream: json.dump(output,stream,indent=2,allow_nan=False)


if __name__=='__main__': main()
