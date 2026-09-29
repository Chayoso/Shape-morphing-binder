"""Current live context, exact confirmed snapshots and raw-head-only promotion."""
from copy import deepcopy

import pytest
import torch

from physmorph.pipeline import post_assimilation_selection as module
from scripts.probes import withdrawal_search_core as driver
from test_window_selection import FakeWithdrawal, equal, make_context


class FakePost(FakeWithdrawal):
    def __init__(self, owner, successor, cfg):
        super().__init__(owner)
        self._next_pins = torch.tensor([True, True, False])

    def evaluate(self, terminal, displacement):
        v = super().evaluate(terminal, displacement)
        v['coast_V'][:, self._next_pins] = 0.
        v['coast_C'][:, self._next_pins] = 0.
        v['coast_pins'] = self._next_pins.clone()
        v['coast_Fp'] = torch.eye(3).repeat(len(self.idx), 1, 1)*1.02
        return v


class Merit:
    def __init__(self, values):
        self.values, self.binding = values, 'live-current-binding'

    def __call__(self, values):
        return deepcopy(self.values)

    def terms(self, values):
        raise AssertionError('The test driver supplies its own search decision')

    def binding_digest(self):
        return self.binding


def make_selection(monkeypatch):
    ctx, original, metrics, owner, lease = make_context(monkeypatch)
    ctx._accepted_velocity = torch.zeros(2, 3, 3)
    ctx._evaluate_merit = Merit(metrics)
    monkeypatch.setattr(module, 'PostAssimilationWindow', FakePost)
    return ctx, original, metrics, owner, lease


def confirmed_driver(monkeypatch, ctx, *, change=None):
    seen = {}

    def run(packet, *, record, raw_observe, successor, cfg, on_confirmed):
        assert packet['rollout'] is ctx._owner and packet['evaluate_merit'] is ctx._evaluate_merit
        assert packet['reference'] is ctx._reference and cfg is ctx._cfg
        assert torch.equal(packet['V'], ctx._accepted_velocity)
        assert packet['lambda_render'] == .2
        assert torch.equal(packet['start_arrived'], torch.tensor([True, False, True]))
        c = torch.zeros(3, 6); c[:, :3] = .1; c[:, 3:] = .2
        values = ctx._model.evaluate(c[:, 3:], c[:, :3])
        report = dict(candidate_found=True, confirmed=True, selected='candidate_h00',
                      merit_binding_unchanged=True)
        info = dict(label='confirm_2', candidate_label='candidate_h00', generation=7,
                    report=deepcopy(report))
        if change is not None:
            change(ctx, c, values, info, report)
        seen.update(values=deepcopy(values), coefficients=c.clone())
        on_confirmed(c, values, info)
        values['positions'].fill_(99); c.zero_()
        return report

    monkeypatch.setattr(driver, 'run_search', run)
    return seen


def test_same_confirmed_head_is_selected_without_another_forward_or_coast_install(monkeypatch):
    ctx, original, _, owner, lease = make_selection(monkeypatch)
    seen = confirmed_driver(monkeypatch, ctx)
    choice, report = ctx.search_post_assimilation(object(), record=lambda *a: None,
                                                 raw_observe=lambda *a: None)
    assert report['confirmed'] and ctx._model.calls == 1
    result, resolved = ctx.resolve(choice)
    assert resolved['selected'] and resolved['raw_certificate']['source_generation'] == 7
    fr, Fs, end, _, hist, stats = result
    torch.testing.assert_close(torch.tensor(fr[-1]), seen['values']['x'], rtol=0, atol=0)
    assert end['v'][1, 0] != 0 and seen['values']['coast_V'][0, 1, 0] == 0
    assert 'Fp' not in end and 'pin' not in end
    equal(hist, original[4])
    assert stats['commit_from_accepted'] is False and stats['mom_out'] is None
    ctx.close()
    assert owner.closed and not lease[0]


def test_inconclusive_search_keeps_exact_original_and_does_not_invent_merit_witness(monkeypatch):
    ctx, original, _, _, _ = make_selection(monkeypatch)
    monkeypatch.setattr(driver, 'run_search', lambda *a, **k:
                        dict(candidate_found=False, status='inconclusive_zero_original_update'))
    choice, report = ctx.search_post_assimilation(object(), record=lambda *a: None,
                                                 raw_observe=lambda *a: None)
    result, resolved = ctx.resolve(choice)
    equal(result, original)
    assert not resolved['selected'] and ctx._model.calls == 0
    assert 'merit_binding_unchanged' not in report


@pytest.mark.parametrize('field', ['coast_V', 'coast_C', 'coast_X', 'coast_Fp', 'coast_pins'])
def test_forged_post_boundary_cannot_be_promoted_even_with_search_success(monkeypatch, field):
    ctx, original, _, _, _ = make_selection(monkeypatch)
    def corrupt(ctx, c, v, info, report):
        if field == 'coast_pins': v[field][1] = False
        elif field == 'coast_Fp': v[field][1, 0, 0] = -1.
        else: v[field][0, 1, 0] = .25
    confirmed_driver(monkeypatch, ctx, change=corrupt)
    choice, _ = ctx.search_post_assimilation(object(), record=lambda *a: None,
                                            raw_observe=lambda *a: None)
    result, report = ctx.resolve(choice)
    assert not report['selected'] and report['failures']
    equal(result, original)


@pytest.mark.parametrize('failure', ['binding', 'unconfirmed', 'wrong_label', 'bad_controls'])
def test_confirmation_identity_and_live_merit_are_mandatory(monkeypatch, failure):
    ctx, _, _, _, _ = make_selection(monkeypatch)
    def corrupt(ctx, c, v, info, report):
        if failure == 'binding': ctx._evaluate_merit.binding = 'changed'
        elif failure == 'unconfirmed': info['report']['confirmed'] = False
        elif failure == 'wrong_label': info['candidate_label'] = 'different'
        else: c.fill_(2.)
    confirmed_driver(monkeypatch, ctx, change=corrupt)
    with pytest.raises(ValueError):
        ctx.search_post_assimilation(object(), record=lambda *a: None, raw_observe=lambda *a: None)


def test_original_merit_and_pace_are_checked_on_last_confirmed_head(monkeypatch):
    ctx, original, metrics, _, _ = make_selection(monkeypatch)
    confirmed_driver(monkeypatch, ctx)
    metrics['merit'] = 10.1
    choice, _ = ctx.search_post_assimilation(object(), record=lambda *a: None,
                                            raw_observe=lambda *a: None)
    result, report = ctx.resolve(choice)
    assert not report['selected'] and 'original_merit_or_pace' in report['failures']
    equal(result, original)


def test_no_saved_merit_or_missing_actual_velocity_can_start_search(monkeypatch):
    ctx, _, _, _, _ = make_selection(monkeypatch)
    ctx._accepted_velocity = None
    with pytest.raises(ValueError, match='accepted V'):
        ctx.search_post_assimilation(object(), record=lambda *a: None, raw_observe=lambda *a: None)
    ctx.close()
    with pytest.raises(RuntimeError, match='expired'):
        ctx.search_post_assimilation(object(), record=lambda *a: None, raw_observe=lambda *a: None)
