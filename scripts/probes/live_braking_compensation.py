"""Fresh callback compensation; no archive admission or production commit."""
from dataclasses import asdict
import itertools
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.probes.braking_compensation import compensate
from scripts.probes.reference_swap import require,sha
from scripts.probes.terminal_braking import Capture,main


def identity_evidence(base):
    folder = base/'work/p303/state_identity1'
    snapshot = base/'work/p303/code_state_identity1'
    report_bytes = (folder/'result.json').read_bytes()
    protocol_bytes = (folder/'protocol.json').read_bytes()
    report = json.loads(report_bytes)
    protocol = json.loads(protocol_bytes)
    json.dumps(report,allow_nan=False)
    require(report['protocol_sha256']==sha256(protocol_bytes).hexdigest(),'Identity protocol mismatch')
    flags = ('exact_warp_inputs','exact_static_inputs_across_forward','owned_C_survives_next_forward',
             'changed_control_changes_raw_C','primal_ownership_pass')
    require(all(report[k] is True for k in flags),'Input/ownership discrimination failed')
    expected = set()
    for control in ('baseline','trial05'):
        labels = [f'{arm}_{control}_{i}' for arm in ('reference','reloaded') for i in range(3)]
        expected.update(itertools.combinations(labels,2))
    pairs = report['comparisons']
    require(len(pairs)==30 and {(r['reference'],r['candidate']) for r in pairs}==expected,'Incomplete pair evidence')
    for row in pairs:
        require(set(row['arrays'])=={'positions','V','F','C'} and
                set(row['gradients'])=={'brake','volume','render'} and
                set(row['data'])=={'volume','silhouette','pbr','render'},'Incomplete witness')
        require(all(0<=v['ratio']<=1 for block in ('arrays','gradients','data')
                    for k,v in row[block].items() if not (block=='arrays' and k=='C')),
                'Failure outside C repeatability; callback alternative is not cleared')
    require(any(r['group']=='within' and r['arrays']['C']['ratio']>1 for r in pairs),
            'No same-instance C failure supporting this alternative protocol')
    archive = base/'work/p303/braking_capture1/owned_window.npz'
    require(sha(archive)==report['archive_sha256']==protocol['archive_sha256'],'Identity input differs')
    bound = {str(folder/name):digest for name,digest in report['sidecars'].items()}
    bound.update({str(snapshot/k):v for k,v in protocol['code'].items()})
    bound.update({str(folder/'result.json'):sha256(report_bytes).hexdigest(),
                  str(folder/'protocol.json'):report['protocol_sha256'],str(archive):report['archive_sha256'],
                  str(snapshot/'scripts/probes/frozen_state_identity.py'):protocol['script_sha256']})
    require(all(sha(Path(k))==v for k,v in bound.items()),'Identity evidence changed')
    return dict(bound=bound,flags={k:report[k] for k in flags},
                original_C_repeat_gate='FAILED; unchanged, not an archive admission',
                failed_pairs=sum(not r['passed'] for r in pairs),
                maximum_C_ratio=max(r['arrays']['C']['ratio'] for r in pairs))


class LiveCompensation(Capture):
    operation = staticmethod(compensate)
    artifact_subdir = 'compensation'
    scope = 'Fresh live callback; same-window feasibility only, no archive reuse or commit'
    rendering_role = 'Prepared CIC/PBR guides the original solve and screens candidates. Compensation minimizes same-ID endpoint error only. Weighted render changes are not causal motion shares or 4K quality.'

    def observe(self,index,packet):
        super().observe(index,packet)
        self.packet = packet
        with np.load(self.out/'trial05.npz',allow_pickle=False) as archive:
            brake = torch.tensor(archive['terminal'],device=packet['x0'].device)
        local = dict(packet,trial05_terminal=brake)
        folder = self.out/self.artifact_subdir
        folder.mkdir(exist_ok=False)
        evidence = {k:v for k,v in local.items() if k not in ('rollout','reference')}
        evidence.update(reference=asdict(packet['reference']),source=self.source,target=self.target)
        packet['rollout'].save(folder/'live_window.npz',evidence)
        self.extra = self.operation(packet['rollout'],local,self.source,self.target,folder)
        self.extra['sidecars']['live_window.npz'] = sha(folder/'live_window.npz')
        self.extra.update(scope=self.scope,rendering_role=self.rendering_role,
                          artifact_subdir=self.artifact_subdir)

    def wrap(self,original):
        wrapped = super().wrap(original)
        def verify(*args,**kwargs):
            result = wrapped(*args,**kwargs)
            if kwargs['win_index']==19:
                require(self.packet.get('optimizer_state_after_callback_exact') is True,
                        'Missing post-callback production isolation proof')
                self.extra['production_state_after_callback_exact'] = True
                require(all(sha(Path(k))==v for k,v in self.identity['bound'].items()),'Identity evidence changed')
            return result
        return verify


if __name__=='__main__':
    LiveCompensation.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    main(LiveCompensation,extra_protocol=dict(kind='Fresh live-window displacement compensation',version=1,
        scope='Noncommitting terminal trial schedule followed by frozen-terminal displacement compensation in one fresh callback; same-window feasibility only',
        identity_discrimination=LiveCompensation.identity,updates=4,halvings=10,
        terminal='Own freshly generated trial05; frozen during compensation',
        admission='No archive load; original C repeat gate remains failed; no production candidate commit'),
        extra_helpers=(Path(__file__).resolve(),Path(__file__).resolve().with_name('braking_compensation.py')))
