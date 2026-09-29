"""Two terminal strengths in one live callback, sharing one immutable baseline."""
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physmorph.pipeline.diagnostic_binding import content_digest
from scripts.probes.running_braking_repair import repair,repair_context
from scripts.probes.live_braking_compensation import LiveCompensation,identity_evidence
from scripts.probes.terminal_braking import main
from scripts.probes.reference_swap import require,sha


def paired_repair(model,packet,source,target,out):
    require('evaluate_merit' in packet,'Paired comparison requires original-merit evaluation')
    terminal_path = out.parent/'trial025.npz'
    terminal_file_digest = sha(terminal_path)
    with np.load(terminal_path,allow_pickle=False) as archive:
        terminal025 = torch.tensor(archive['terminal'],device=model.coefficients.device)
    choices = (('terminal05',packet['trial05_terminal'].detach().clone()),('terminal025',terminal025))
    choice_hashes = {label:content_digest(coeff) for label,coeff in choices}
    folder = out/'baseline';folder.mkdir(exist_ok=False)
    baseline = repair(model,packet,source,target,folder,baseline_only=True)
    baseline_digest = baseline.digest()
    shared = baseline.decode()
    metadata = dict(**shared,context_sha256=baseline.context,package_sha256=baseline_digest,
                    arrays={key:dict(dtype=dtype,shape=shape,sha256=sha256(data).hexdigest())
                            for key,dtype,shape,data in baseline.arrays})
    manifest = folder/'shared_baseline.json'
    manifest.write_text(json.dumps(metadata,indent=2,allow_nan=False))
    sidecars = {'baseline/'+k:v for k,v in shared['sidecars'].items()}
    sidecars['baseline/shared_baseline.json'] = sha(manifest)
    arms = {}
    for label,terminal in choices:
        require(baseline.digest()==baseline_digest,'Shared baseline package changed before arm')
        require(repair_context(model,packet,source,target)==baseline.context,'Paired start context changed')
        require(content_digest(terminal)==choice_hashes[label],'Selected terminal changed before arm')
        folder = out/label;folder.mkdir(exist_ok=False)
        arm = repair(model,packet,source,target,folder,correction_rounds=2,quality_backtracking=True,
                     shared_baseline=baseline,selected_terminal=terminal,terminal_label=label)
        require(baseline.digest()==baseline_digest and repair_context(model,packet,source,target)==baseline.context,
                'Shared baseline/context changed after arm')
        require(content_digest(terminal)==choice_hashes[label],'Selected terminal changed after arm')
        arm.update(selected_terminal_sha256=choice_hashes[label],shared_context_before_after_exact=True)
        arms[label] = arm
        sidecars.update({label+'/'+k:v for k,v in arm['sidecars'].items()})
        (folder/'arm_result.json').write_text(json.dumps(arm,indent=2,allow_nan=False))
        sidecars[label+'/arm_result.json'] = sha(folder/'arm_result.json')
        # Always retain both arms; no selection by report-only original merit.
    require(sha(terminal_path)==terminal_file_digest,'Preliminary terminal artifact changed')
    return dict(baseline=metadata,arms=arms,sidecars=sidecars,
                arm_order=[label for label,_ in choices],terminal025_file_sha256=terminal_file_digest,
                original_merit_role='Report only; fixed maximum of shared original three replay merits, no added closure slack; not an Armijo or outer-acceptance certificate')


class PairedRepair(LiveCompensation):
    operation = staticmethod(paired_repair)
    record_candidate_merit = True
    artifact_subdir = 'paired_repair'
    scope = 'Fresh callback paired terminal05/025 with one immutable baseline and original-merit reporting; no archive admission or commit'
    rendering_role = 'Same prepared CIC/PBR references and lambda constrain both arms; full original merit and individual render components are separate reports, not causal direction shares or 4K quality'


if __name__=='__main__':
    PairedRepair.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    main(PairedRepair,extra_protocol=dict(kind='Paired terminal strengths on one live window',version=1,
        scope=PairedRepair.scope,identity_discrimination=PairedRepair.identity,
        arms=['terminal05','terminal025'],shared_baseline_repeats=3,
        per_arm=dict(halvings=10,correction_rounds=2,accepted_updates=1,fixed_candidate_repeats=3,running_noise_repeats=3),
        baseline='Immutable byte package; one prepared data/P306/original-merit reference; original controls/spec/reference/cohorts/lambda/wu bound before/after each arm',
        merit='Original shared physics+lambda*render from same-forward x/F/v/V/body energy; individual original baseline32eps closure; report-only nonincrease at max of same three baseline merits with no added slack',
        stopping='Each arm uses unchanged P310 gates and stops after first accepted candidate plus3fixed repeats even if merit or any repeat fails; always run both arms',
        admission='No old archive admission or production commit; raw metrics filter selection, not independent post-selection validation'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('paired_braking_repair.py','running_braking_repair.py','live_braking_compensation.py','braking_compensation.py')))
