"""Immediate, scoped selection of an owned private body-control forward.

This context never changes optimizer history, installs controls, or certifies
raw quality. Its trusted caller supplies the raw decision for a registered
same-forward choice; ordinary runner admission remains a separate operation.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import torch

from ..compute import to_array
from .endpoint_contract import endpoint_bounds, valid_endpoint
from .frozen_body_window import FrozenBodyWindow
from .frozen_withdrawal_window import FrozenWithdrawalWindow
from .prepared_reference import PreparedReference


def validate_config(cfg, *, allow_auto=False):
    unsupported = [name for name in (
        'commit_pic', 'commit_pic_objective', 'shift_sub', 'reattach', 'rest_commit',
        'rest_commit_reversal', 'settle_commit', 'settle_pin_follow', 'settle_pin_yield',
        'geometric_rest', 'geometric_variance', 'render_F_geom', 'use_gauss_loss',
        'surface_gs_loss', 'continuity', 'settle_pin_kkt', 'opt_material', 'grad_dump',
        'lg_sweeps', 'local_dress_iters') if getattr(cfg, name, False)]
    if (not cfg.body_ctrl or not cfg.body_terminal_ctrl or cfg.T < 2 or cfg.mom_carry != 0
            or cfg.phys_loss not in (('ot_pace', 'auto') if allow_auto else ('ot_pace',))
            or cfg.loss_units != 'density'):
        unsupported.append('raw fixed-material density ot_pace two-mode body / mom_carry=0')
    if unsupported:
        raise ValueError('Prepared window selection does not support '+', '.join(unsupported))


def _own(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _own(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_own(item) for item in value)
    return deepcopy(value)


class PreparedWindowSelection:
    """Opaque choices live only until close; exposed inspections own their data."""

    def __init__(self, original, owner, reference, evaluate_merit, merit_lease, cfg, prm, win_index):
        self._closed, self._model, self._choices = False, None, {}
        self._owner, self._lease = owner, merit_lease
        try:
            self._initialize(original, owner, reference, evaluate_merit, merit_lease, cfg, prm, win_index)
        except Exception:
            self.close()
            raise

    def _initialize(self, original, owner, reference, evaluate_merit, merit_lease, cfg, prm, win_index):
        validate_config(cfg)
        if (not isinstance(owner, FrozenBodyWindow) or owner.closed or not owner.spec.body_ctrl
                or owner.spec.body_modes != 2 or owner.spec.T != cfg.T
                or asdict(owner.spec.prm) != asdict(prm)):
            raise ValueError('Selection requires the live final two-mode rollout owner')
        if not isinstance(reference, PreparedReference) or reference.render is None:
            raise ValueError('Selection requires the frozen CIC/PBR prepared reference')
        if (not isinstance(original, tuple) or len(original) != 6 or not original[4]
                or original[5].get('pace_bound') or not original[5].get('accepted', 0)):
            raise ValueError('Selection requires an accepted original without a pace-bound decision')
        if not callable(evaluate_merit) or not isinstance(merit_lease, list) or merit_lease != [True]:
            raise ValueError('Selection requires its live final merit evaluator')
        self._cfg, self._prm, self._reference = deepcopy(cfg), deepcopy(prm), deepcopy(reference)
        self._owner, self._evaluate_merit, self._lease = owner, evaluate_merit, merit_lease
        self._original = _own(original)
        self._model = FrozenWithdrawalWindow(owner)
        self._start = torch.as_tensor(self._original[0][0], device=owner.coefficients.device).detach().clone()
        self._pins = torch.as_tensor(self._model.spec.pin, device=self._start.device) > .5 if self._model.spec.pin is not None else torch.zeros(len(self._start), device=self._start.device, dtype=torch.bool)
        self._loss = float(self._original[4][-1]['loss'])
        start_loss = self._original[5].get('L_start')
        if not math.isfinite(self._loss) or (cfg.pace > 0 and (start_loss is None or not math.isfinite(start_loss))):
            raise ValueError('Selection requires finite original merit and active pace reference')
        if (len(original[0]) != cfg.T+1 or len(original[1]) != cfg.T+1
                or not torch.equal(self._start, torch.as_tensor(self._model.spec.x0, device=self._start.device))):
            raise ValueError('Original trajectory does not match its prepared rollout start')
        self._ceiling = self._loss + 32*torch.finfo(self._start.dtype).eps*max(abs(self._loss), 1e-12)
        self._floor = (1-cfg.pace)*float(start_loss) if cfg.pace > 0 else -math.inf
        self._window, self._closed = int(win_index), False
        self._choices, self._labels = {}, set()
        self._identity = object()
        fr, Fs, end, _, hist, _ = self._original
        tensor = lambda value: torch.as_tensor(value, device=self._start.device).detach().clone()
        self._choices[self._identity] = dict(label='original', identity=True, eligible=True, failures=[],
            coefficients=self._model.coefficients.clone(), metrics=_own(hist[-1]), certificate=None,
            values=dict(x=tensor(fr[-1]), F=tensor(end['F']).reshape(-1, 9), v=tensor(end['v']),
                        C=tensor(end['C']), positions=torch.stack([tensor(x) for x in fr[1:]]),
                        F_sequence=torch.stack([tensor(x).reshape(-1, 3, 3) for x in Fs[1:]]),
                        F_initial=tensor(Fs[0]).reshape(-1, 3, 3)), result=self._original)

    def _live(self):
        if self._closed or not self._lease[0] or self._owner.closed:
            raise RuntimeError('Prepared window selection expired')

    def _get(self, choice):
        self._live()
        if choice not in self._choices:
            raise ValueError('Foreign or unregistered window choice')
        return self._choices[choice]

    def original(self):
        self._live()
        return self._identity

    @property
    def closed(self):
        return self._closed

    def evaluate(self, coefficients, label):
        self._live()
        if not isinstance(label, str) or not label or label == 'original' or label in self._labels:
            raise ValueError('Candidate label must be distinct and nonempty')
        self._labels.add(label)
        choice = object()
        entry = dict(label=label, identity=False, eligible=False, failures=[], certificate=None,
                     values=None, metrics=None, coefficients=None, result=None)
        self._choices[choice] = entry
        expected = self._model.coefficients
        if (not torch.is_tensor(coefficients) or coefficients.shape != expected.shape
                or coefficients.dtype != expected.dtype or coefficients.device != expected.device
                or not bool(torch.isfinite(coefficients).all())
                or bool((coefficients.square().sum(-1) > 1+1e-6).any())):
            entry['failures'].append('invalid_joint_coefficients')
            return choice
        entry['coefficients'] = coefficients.detach().clone()
        with torch.no_grad():
            try:
                values = self._model.evaluate(entry['coefficients'][:, 3:], entry['coefficients'][:, :3])
                entry['values'] = _own(values)
                failures, post_det = self._health(entry['values'])
                entry['failures'].extend(failures)
                if failures:
                    return choice
                metrics = self._evaluate_merit(entry['values'])
                entry['metrics'] = _own(metrics)
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    entry['failures'].append('nonfinite_merit')
                elif not self._floor <= metrics['merit'] <= self._ceiling:
                    entry['failures'].append('original_merit_or_pace')
                else:
                    entry['result'] = self._candidate_result(entry, post_det)
                    entry['eligible'] = True
            except ValueError as error:
                entry['failures'].append('invalid_private_forward: '+str(error))
        return choice

    def _health(self, v):
        n, T = len(self._start), self._cfg.T
        failures = []
        required = dict(x=(n, 3), F=(n, 9), v=(n, 3), C=(n, 3, 3), positions=(T, n, 3),
            V=(T, n, 3), F_initial=(n, 3, 3), F_sequence=(T, n, 3, 3),
            C_sequence=(T, n, 3, 3), coast_X=(T+1, n, 3), coast_V=(T+1, n, 3),
            coast_F=(T+1, n, 9), coast_C=(T+1, n, 3, 3))
        if any(key not in v or not torch.is_tensor(v[key]) or tuple(v[key].shape) != shape
               or v[key].device != self._start.device or v[key].dtype != self._start.dtype for key, shape in required.items()):
            return ['invalid_full_state_layout'], None
        if not all(bool(torch.isfinite(value).all()) for value in v.values() if torch.is_tensor(value)):
            return ['nonfinite_full_state'], None
        post = torch.linalg.det(v['F_sequence'])
        pre = torch.cat((v['F_initial'][None], v['F_sequence'][:-1]))
        effective = torch.linalg.det(pre+self._model.stress.reshape(T, n, 3, 3))
        if float(torch.minimum(post.min(), effective.min())) <= 1e-4:
            failures.append('post_or_effective_determinant')
        if not (v.get('valid') and v.get('pins_exact')) or float(torch.linalg.det(v['coast_F'].reshape(-1, 3, 3)).min()) <= 0:
            failures.append('invalid_private_health')
        bounds = endpoint_bounds(self._prm, self._start)
        if not valid_endpoint(v['positions'], bounds) or not valid_endpoint(v['coast_X'], bounds):
            failures.append('all_step_bounds')
        initial_F = torch.as_tensor(self._original[1][0], device=self._start.device).reshape(n, 3, 3)
        equalities = ((v['x'], v['positions'][-1]), (v['v'], v['V'][-1]),
            (v['F'], v['F_sequence'][-1].reshape(n, 9)), (v['C'], v['C_sequence'][-1]),
            (v['F_initial'], initial_F), (v['coast_X'][0], v['x']), (v['coast_V'][0], v['v']),
            (v['coast_F'][0], v['F']), (v['coast_C'][0], v['C']))
        if not all(torch.equal(a, b) for a, b in equalities):
            failures.append('same_forward_boundary')
        pinned = self._pins
        if not (torch.equal(v['positions'][:, pinned], self._start[pinned][None].expand(T, -1, -1))
                and torch.equal(v['coast_X'][:, pinned], self._start[pinned][None].expand(T+1, -1, -1))
                and all(bool((a == 0).all()) for a in (v['V'][:, pinned], v['C_sequence'][:, pinned],
                                                       v['coast_V'][1:, pinned], v['coast_C'][1:, pinned]))):
            failures.append('exact_pins')
        return failures, post

    def _candidate_result(self, entry, post_det):
        v, m = entry['values'], entry['metrics']
        _, _, _, material, history, stats = _own(self._original)
        fr = [to_array(self._start, copy=True)]+[to_array(x, copy=True) for x in v['positions']]
        Fs = [to_array(v['F_initial'], copy=True)]+[to_array(F, copy=True) for F in v['F_sequence']]
        end = dict(F=to_array(v['F'].reshape(-1, 3, 3), copy=True), v=to_array(v['v'], copy=True),
            C=to_array(v['C'], copy=True), Fg=None, n_inv_steps=int((post_det <= 0).any(0).sum()),
            Jmin_traj=float(post_det.min()))
        donor = history[-1]
        render = m['render'] if donor.get('d_render') is not None else None
        pbr = m['pbr'] if donor.get('d_pbr') is not None else None
        observation = dict(loss=m['merit'], d_vol=m['volume'], kin=m['stored_terminal'],
            kin_run=m['stored_running'], kin_var=m['stored_variance'], d_pbr=pbr,
            d_render=None if render is None else render-self._cfg.w_pbr*(pbr or 0.),
            d_render_total=render, d_sil=m['silhouette'] if render is not None else None,
            d_gauss=None, **{'lambda': m['lambda_render'] if render is not None else None},
            dfc_absmax=donor.get('dfc_absmax'), s_absmax=donor.get('s_absmax'),
            endpoint_space='raw_rollout', observation_scope='fresh private selected forward; donor optimizer history unchanged')
        clear = ('grad_norm', 'alpha', 'predicted_decrease', 'step_norm', 'render_cos', 'phys_cos',
                 'render_work', 'render_work_x', 'render_work_F', 'phys_work', 'phys_work_x',
                 'phys_work_F', 'phys_work_v', 'g_cos', 'g_raw_cos', 'g_share', 'g_phys_norm', 'g_rend_norm',
                 'render_channels', 'h1_ratio')
        for key in clear:
            observation[key] = None
            if key in stats:
                stats[key] = None
        stats.update(selected_observation=observation, commit_from_accepted=False, mom_out=None,
            motion_accounting=None, gx=None, owned_endpoint=None,
            replay_diagnostics=dict(commit_source='private_selected_forward', position_space='raw_rollout',
                                    replay_E_final=m['merit'], replay_lambda_final=m['lambda_render']))
        coeff = entry['coefficients']
        field = self._prm.dx*(coeff[self._model.idx]*self._model.weights[..., None]).sum(1)*self._model.gate
        modes = field.reshape(len(field), 2, 3)
        stats.update(body_rms_wu=float(modes[:, 0].square().sum(1).mean().sqrt()),
            body_terminal_rms_wu=float(modes[:, 1].square().sum(1).mean().sqrt()),
            body_coeff_max=float(coeff.norm(dim=1).max()),
            body_coeff_saturated_frac=float((coeff.norm(dim=1) >= .999).float().mean()))
        return fr, Fs, end, material, history, stats

    def inspect(self, choice):
        entry = self._get(choice)
        return dict(window=self._window, **_own({key: entry[key] for key in (
            'label', 'identity', 'eligible', 'failures', 'values', 'metrics', 'coefficients', 'certificate')}))

    def certify(self, choice, report):
        """Trusted caller binding only; this API does not validate raw-quality evidence."""
        entry = self._get(choice)
        if not isinstance(report, dict) or type(report.get('passed')) is not bool:
            raise ValueError('Raw report requires an explicit boolean passed decision')
        entry['certificate'] = _own(report)

    def resolve(self, choice=None):
        entry = self._get(self._identity if choice is None else choice)
        certified = entry['certificate'] is not None and entry['certificate']['passed'] is True
        selected = not entry['identity'] and entry['eligible'] and certified
        report = dict(window=self._window, requested_label=entry['label'], selected=selected,
            selected_label=entry['label'] if selected else 'original', failures=list(entry['failures']),
            raw_certificate=_own(entry['certificate']),
            certificate_scope='Trusted caller report bound to this owned evaluation; context does not certify physical quality',
            merit_ceiling=self._ceiling, pace_floor=self._floor if math.isfinite(self._floor) else None,
            donor_history_scope='Unchanged Adam history, accepted/rejected counts, render_influence_steps and body update telemetry describe donor optimization only',
            selected_work_scope='No gradient/work or Adam-step claim for the selected private forward',
            donor_observation=_own(self._original[4][-1]),
            donor_render_channels=_own(self._original[5].get('render_channels')),
            donor_replay_diagnostics=_own(self._original[5].get('replay_diagnostics')))
        if not entry['identity'] and not certified:
            report['failures'].append('missing_or_failed_raw_certificate')
        return _own(entry['result'] if selected else self._original), report

    def close(self):
        if not self._closed:
            self._closed = True
            if isinstance(self._lease, list) and self._lease:
                self._lease[0] = False
            if self._model is not None:
                self._model.close()
            if isinstance(self._owner, FrozenBodyWindow):
                self._owner.close()
            self._choices.clear()
            for name in ('_original', '_model', '_owner', '_evaluate_merit', '_reference',
                         '_cfg', '_prm', '_start', '_pins', '_lease'):
                setattr(self, name, None)
