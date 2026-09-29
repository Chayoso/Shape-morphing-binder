"""Capture one fresh W20 realization and verify reusable GPU rollout/reference state."""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.prepared_reference import PreparedReference
from scripts.probes.terminal_braking import Capture,main
from scripts.probes.reference_swap import require,sha


def signature(model,reference,terminal,mask):
    with torch.enable_grad():
        leaf = terminal.detach().clone().requires_grad_()
        values = model.evaluate(leaf)
        require(values['valid'],'Invalid serialization witness')
        geometric = (values['positions'][-1]-values['positions'][-2])/model.spec.prm.dt
        h = .5*(values['v'][mask].square().sum(-1).mean()+geometric[mask].square().sum(-1).mean())
        terms = reference.terms(values['x'])
        gradients = {key:torch.autograd.grad(value,leaf,retain_graph=key!='render')[0].detach()
                     for key,value in [('brake',h),('volume',terms['volume']),('render',terms['render'])]}
    return dict(arrays={k:values[k].detach().clone() for k in ('positions','V','F','C')},
                terms={k:float(v.detach()) for k,v in terms.items()},gradients=gradients)


class OwnedCapture(Capture):
    def observe(self,index,packet):
        super().observe(index,packet)
        model = packet['rollout']
        archive = self.out/'owned_window.npz'
        observations = {k:v for k,v in packet.items() if k not in ('rollout','reference')}
        observations.update(reference=asdict(packet['reference']),source=self.source,target=self.target)
        with np.load(self.out/'trial05.npz',allow_pickle=False) as data:
            brake = torch.tensor(data['terminal'],device=packet['x0'].device)
        observations['trial05_terminal'] = brake
        model.save(archive,observations)
        digest = sha(archive)
        arrived = packet['start_arrived'] & ~packet['pins']
        controls = [packet['controls']['body'][:,3:],brake]
        original = [signature(model,packet['reference'],b,arrived) for b in controls]
        model.close()
        restored,loaded = FrozenBodyWindow.load(archive,str(packet['x0'].device))
        try:
            require(torch.equal(loaded['pins'],packet['pins']) and
                    torch.equal(loaded['start_arrived'],packet['start_arrived']), 'Reloaded cohorts differ')
            reference = PreparedReference(**loaded['reference'])
            loaded_controls = [loaded['controls']['body'][:,3:],loaded['trial05_terminal']]
            require(all(torch.equal(a,b) for a,b in zip(controls,loaded_controls)), 'Reloaded witness controls differ')
            loaded_arrived = loaded['start_arrived'] & ~loaded['pins']
            replay = [signature(restored,reference,b,loaded_arrived) for b in loaded_controls]
            eps = torch.finfo(brake.dtype).eps
            closure = []
            for old,new in zip(original,replay):
                checks = {}
                for key,unit in (('positions',restored.spec.prm.dx),
                                 ('V',restored.spec.prm.dx/(restored.spec.T*restored.spec.prm.dt)),('F',1.),
                                 ('C',1/(restored.spec.T*restored.spec.prm.dt))):
                    error = (old['arrays'][key]-new['arrays'][key]).abs()
                    tolerance = 32*eps*(unit+old['arrays'][key].abs())
                    checks[key] = float((error/tolerance).max())
                    require(checks[key]<=1.,'Reloaded trajectory differs: '+key)
                for key,g in old['gradients'].items():
                    error = float((g-new['gradients'][key]).norm())
                    tolerance = 64*eps*max(float(g.norm()),1e-12)
                    checks['gradient_'+key] = error/tolerance
                    checks['gradient_detail_'+key] = dict(error_norm=error,reference_norm=float(g.norm()),tolerance=tolerance)
                    require(error<=tolerance,'Reloaded derivative differs: '+key)
                for key,value in old['terms'].items():
                    tolerance = 32*eps*max(abs(value),1e-12)
                    checks['data_'+key] = abs(value-new['terms'][key])/tolerance
                    require(checks['data_'+key]<=1.,'Reloaded reference differs: '+key)
                closure.append(checks)
        finally:
            restored.close()
        require(sha(archive)==digest,'Owned archive changed during validation')
        self.sidecars[archive.name] = digest
        self.extra = dict(archive_sha256=digest,roundtrip_closure=closure,
            scope='Fresh realization, not reconstruction of terminal_braking3; baseline and its own trial05, no compensation yet')


if __name__=='__main__':
    main(OwnedCapture,extra_protocol=dict(kind='Reusable frozen-window capture',version=1,
         witnesses=['baseline','fresh_trial05'],no_pickle=True,compensation=False,
         array_tolerance_eps=32,gradient_relative_tolerance_eps=64),
         extra_helpers=(Path(__file__).resolve(),))
