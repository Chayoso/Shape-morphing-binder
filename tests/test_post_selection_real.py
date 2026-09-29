"""Actual MPM/search/registration, with explicitly synthetic data constraints.

This isolates state delivery and new-pin handling; it is not a shape-quality gate.
"""
import torch

from physmorph.compute import to_array
from physmorph.pipeline.prepared_reference import PreparedReference
from physmorph.pipeline.window_selection import PreparedWindowSelection
from test_post_assimilation_window import make_case


class ConstantReference(PreparedReference):
    def terms(self, x):
        z = x.sum()*0
        return dict(volume=z, render=z, silhouette=z, pbr=z)


class ConstantMerit:
    def __call__(self, values):
        return dict(merit=1., physical=1., render=0., volume=0., silhouette=0., pbr=0.,
                    stored_terminal=0., stored_running=0., stored_variance=0.,
                    body_energy=float(values['body_energy'].detach()), lambda_render=.3)

    def terms(self, values):
        return dict(merit=1+values['positions'].sum()*0)

    def binding_digest(self):
        return 'constant-data-fixture-not-native-merit'


def build_context(owner, cfg):
    cfg.pace = 0.
    base = owner.evaluate(owner.coefficients[:, 3:], retain_full_state=True)
    array = lambda x: to_array(x, copy=True)
    frames = [array(owner.spec.x0)]+[array(x) for x in base['positions']]
    Fs = [array(base['F_initial'])]+[array(x) for x in base['F_sequence']]
    end = dict(F=array(base['F'].reshape(-1, 3, 3)), v=array(base['v']), C=array(base['C']),
               Fg=None, n_inv_steps=0, Jmin_traj=float(torch.linalg.det(base['F_sequence']).min()))
    history = [dict(loss=1., d_render=0., d_pbr=0., d_vol=0., d_sil=0.,
                    body_update_modes_rms=[.01, .01], **{'lambda': .3})]
    stats = dict(accepted=1, pace_bound=False, commit_from_accepted=True, mom_out=None,
                 arrived_mask=torch.ones(len(frames[0]), dtype=torch.bool, device=base['x'].device))
    original = frames, Fs, end, None, history, stats
    context = PreparedWindowSelection(original, owner, ConstantReference({}, {}, {}, .3, 'fixture'),
        ConstantMerit(), [True], cfg, owner.spec.prm, 0, accepted_velocity=base['V'])
    return context, original


def exercise_selection(owner, successor, cfg):
    ctx, original = build_context(owner, cfg)
    rows, baselines = {}, []
    def raw(label, values, baseline=False):
        if baseline:
            baselines.append(label)
        return dict(passed=True, baseline_ready=len(baselines) == 3,
                    scope='State integration only; raw shape constraint is a stub')
    def record(label, values, info):
        rows[label] = {key: value.detach().clone() for key, value in values.items()
                       if torch.is_tensor(value)}
    try:
        choice, report = ctx.search_post_assimilation(successor, record=record, raw_observe=raw)
        assert report['candidate_found'] and report['confirmed'], report
        assert [row['label'] for row in report['baselines']] == ['baseline_0', 'baseline_1', 'baseline_2']
        assert [row['label'] for row in report['confirmations']] == ['confirm_0', 'confirm_1', 'confirm_2']
        assert all(row['passed'] for row in report['baselines']+report['confirmations'])
        actual, resolved = ctx.resolve(choice)
        assert resolved['selected'], resolved
        last = rows['confirm_2']
        device = last['x'].device
        as_tensor = lambda x: torch.as_tensor(x, device=device)
        assert torch.equal(torch.stack([as_tensor(x) for x in actual[0][1:]]), last['positions'])
        assert torch.equal(torch.stack([as_tensor(x) for x in actual[1][1:]]), last['F_sequence'])
        assert torch.equal(as_tensor(actual[0][-1]), last['positions'][-1])
        assert torch.equal(as_tensor(actual[2]['F']).reshape_as(last['F']), last['F'])
        assert torch.equal(as_tensor(actual[2]['v']), last['V'][-1])
        assert torch.equal(as_tensor(actual[2]['C']), last['C_sequence'][-1])
        new = last['coast_pins'] & ~ctx._pins
        assert bool(new.any()) and bool((last['v'][new] != 0).any())
        assert bool((last['coast_V'][:, new] == 0).all())
        assert bool((last['coast_C'][:, new] == 0).all())
        assert 'Fp' not in actual[2] and 'pin' not in actual[2]
        assert actual[4] == original[4]
        # Registration is lazy: only the search model executed numerical forwards.
        assert ctx._model.adjoint is None
        assert report['cohorts']['surviving_free'] == int((~last['coast_pins']).sum())
        return {key: report[key] for key in ('status', 'cohorts', 'baseline_objective_min',
                                            'gradient_norm', 'selected', 'confirmations')}
    finally:
        ctx.close()


def test_actual_cpu_search_registers_only_last_confirmed_raw_head():
    owner, successor, cfg = make_case(fp64=True)
    exercise_selection(owner, successor, cfg)
