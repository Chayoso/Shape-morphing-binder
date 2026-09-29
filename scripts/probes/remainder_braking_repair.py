"""Bounded observed-model correction in a fresh, noncommitting W20 callback."""
from functools import partial
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.compute import cuda_execution
from physmorph.pipeline.affine_braking import affine_ball_step,observed_remainder
from scripts.probes.reference_swap import require
from scripts.probes.terminal_braking import main
from scripts.probes.running_braking_repair import RunningRepair,repair
from scripts.probes.live_braking_compensation import identity_evidence


class RemainderRepair(RunningRepair):
    operation = staticmethod(partial(repair,correction_rounds=2))
    artifact_subdir = 'remainder_repair'
    scope = 'Fresh live callback; bounded observed-model-remainder correction only, no archive reuse or commit'


if __name__=='__main__':
    RemainderRepair.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    with cuda_execution('cuda:0'):
        g = torch.tensor([0.,-1.],dtype=torch.float64,device='cuda:0')
        G = torch.tensor([[1.,0.],[-1.,1e-4]],dtype=torch.float64,device=g.device)
        b = torch.tensor([-.6,.6-.8e-4],dtype=g.dtype,device=g.device)
        step,info = affine_ball_step(g,G,b,1.)
        require(step is not None and bool(torch.allclose(step,g.new_tensor([-.6,-.8]),rtol=0,atol=2e-8)),
                'CUDA affine smoke failed')
        e = observed_remainder(G@step+g.new_tensor([.01,.02]),torch.zeros_like(b),G,step)
        require(bool(torch.allclose(e,g.new_tensor([.01,.02]),rtol=0,atol=1e-12)),'CUDA remainder smoke failed')
        print(json.dumps(dict(cuda_remainder_smoke=True,linear=info)),flush=True)
    main(RemainderRepair,extra_protocol=dict(kind='Fresh live-window observed-model-remainder repair',version=1,
        scope=RemainderRepair.scope,identity_discrimination=RemainderRepair.identity,updates=4,halvings=10,
        correction_rounds=2,fixed_candidate_repeats=3,
        objective='Mean squared geometric velocity across all physical steps and fixed start-arrived-free IDs',
        correction='At each halving freeze origin/Jacobian/trust. Replace e by actual-origin-G*final projected delta; solve RHS=original ceilings-origin-e. Never accumulate e or advance a rejected origin.',
        resets='Each halving starts e=0; only an accepted displacement update establishes a new origin',
        stopping='First accepted running-repair candidate passing all P306 gates is replayed three times against frozen original baselines; stop even if any repeat fails',
        data_constraints='Original max-of-three prepared volume/render ceilings unchanged; shifted-model and original affine checks logged separately',
        terminal='Own freshly generated trial05, fixed',
        admission='No archive admission or production candidate commit; no curvature or persistence claim'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('remainder_braking_repair.py','running_braking_repair.py','live_braking_compensation.py','braking_compensation.py')))
