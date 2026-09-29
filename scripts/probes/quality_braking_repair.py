"""Keep the displacement origin when a data-restored repair fails raw quality."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.probes.terminal_braking import main
from scripts.probes.running_braking_repair import repair
from scripts.probes.remainder_braking_repair import RemainderRepair
from scripts.probes.live_braking_compensation import identity_evidence


class QualityRepair(RemainderRepair):
    operation = staticmethod(partial(repair,correction_rounds=2,quality_backtracking=True))
    artifact_subdir = 'quality_repair'
    scope = 'Fresh live callback; unchanged raw-quality gates also filter backtracking, no archive reuse or commit'


if __name__=='__main__':
    QualityRepair.identity = identity_evidence(Path('/data/relcfd/chayo/physmorph_v2'))
    main(QualityRepair,extra_protocol=dict(kind='Fresh live-window quality-filtered running repair',version=1,
        scope=QualityRepair.scope,identity_discrimination=QualityRepair.identity,
        quality_backtracking=True,updates=1,halvings=10,correction_rounds=2,fixed_candidate_repeats=3,
        objective='Same P309 geometric running objective and affine prepared volume/render constraints',
        correction='Same P309 frozen-origin, replaced observed-model remainder; actual nonlinear ceilings unchanged',
        acceptance='Require data, resolved running decrease and every unchanged P306 raw/motion gate before advancing origin. If data passes but quality fails, log rejection and try next smaller radius from the same origin.',
        stopping='First accepted all-gate running repair gets three fixed-coefficient replays; stop even if a replay fails',
        terminal='Own freshly generated trial05, fixed',
        admission='No archive admission or production candidate commit; raw metrics filter selection without differentiation, not independent post-selection validation'),
        extra_helpers=tuple(Path(__file__).resolve().with_name(name) for name in
            ('quality_braking_repair.py','remainder_braking_repair.py','running_braking_repair.py',
             'live_braking_compensation.py','braking_compensation.py')))
