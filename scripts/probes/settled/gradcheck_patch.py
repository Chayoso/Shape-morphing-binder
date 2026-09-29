"""Patch a COPY of this branch's optimizer with a first-iteration diagnostic hook (the render-off twin is the
built-in --render_weight_scale 0). Nothing else changes.
  MJ_GRADCHECK=path   at window MJ_GRADCHECK_WIN (default 1), first iteration: finite-difference check of the physics,
                      render and cleanup gradients along several directions, render-influence telemetry, and a
                      momentum budget of the rollout; a JSON report is written to path and the process exits.
Usage on hyde06 (on a COPY of the checkout, never the working one):
  cp -r $REPO $REPO_dbg && $PY scripts/probes/settled/gradcheck_patch.py $REPO_dbg/physmorph/pipeline/optimizer.py
  MJ_GRADCHECK=$OUT/gc_w20.json MJ_GRADCHECK_WIN=20 <the run's command from $REPO_dbg>
  $PY scripts/probes/settled/gc_read.py $OUT/gc_w20.json [lambda]; $PY scripts/probes/settled/gc_mom.py $OUT/gc_w20.json
(At window 1 the hook runs before the balancer sets lambda; pass the run's calibrated lambda to gc_read.)"""
import sys, re
p = sys.argv[1]
s = open(p).read()
n = 0
anchor = "            if cfg.grad_dump and it == 0 and lr is not None:\n"
assert s.count(anchor) == 1
hook = r'''            if it == 0 and os.environ.get("MJ_GRADCHECK"):
                tgt._gc_calls = getattr(tgt, "_gc_calls", 0) + 1
            if it == 0 and os.environ.get("MJ_GRADCHECK") and tgt._gc_calls == int(os.environ.get("MJ_GRADCHECK_WIN", "1")):
                import json as _json
                _rep = {"window": int(tgt._gc_calls), "N": int(N), "rollout_steps": int(T), "control_steps": int(cfg.T),
                        "settled": bool(use_settled), "lam_r": float(lam_r), "lambda_scale": float(cfg.render_weight_scale)}
                _gr_true = torch.autograd.grad(lr, leaves, retain_graph=True, allow_unused=True)
                _gr_true = [torch.zeros_like(l) if g_ is None else g_.detach() for l, g_ in zip(leaves, _gr_true)]
                _gp = [g_.detach() for g_ in gp]
                _gd = [g_.detach() for g_ in gdt] if gdt is not None else None
                _names = ["dFc"] + (["s"] if s is not None else []) + (["u"] if u is not None else [])
                with torch.no_grad():
                    _dc0 = expand(dFc.detach())
                    _rep["release_dfc_absmax"] = float(_dc0[cfg.T:].abs().max()) if use_settled else None
                    _rep["driven_dfc_absmax"] = float(_dc0[:cfg.T].abs().max())
                # render influence: per leaf and total, raw and after the one-sided PCGrad projection
                def _dot(a_, b_):
                    return float(sum((x_ * y_).sum() for x_, y_ in zip(a_, b_)))
                def _nrm(a_):
                    return float(sum((x_ * x_).sum() for x_ in a_)) ** 0.5
                _dpr = _dot(_gp, _gr_true)
                _gr_proj = ([r_ - (_dpr / max(_nrm(_gp) ** 2, 1e-30)) * p_ for r_, p_ in zip(_gr_true, _gp)]
                            if _dpr < 0 else _gr_true)
                _infl = {"total": {"|g_phys|": _nrm(_gp), "|g_render_raw|": _nrm(_gr_true), "|g_render_proj|": _nrm(_gr_proj),
                                   "cos_raw": _dpr / max(_nrm(_gp) * _nrm(_gr_true), 1e-30),
                                   "projected": bool(_dpr < 0),
                                   "g_share": float(lam_r) * _nrm(_gr_proj) / max(_nrm(_gp) + float(lam_r) * _nrm(_gr_proj), 1e-30),
                                   "|g_cleanup|": _nrm(_gd) if _gd is not None else 0.0}}
                for i_, nm_ in enumerate(_names):
                    a_, b_ = _gp[i_], _gr_true[i_]; c_ = _gr_proj[i_]
                    _infl[nm_] = {"|g_phys|": float(a_.norm()), "|g_render_raw|": float(b_.norm()), "|g_render_proj|": float(c_.norm()),
                                  "cos_raw": float((a_ * b_).sum() / (a_.norm() * b_.norm()).clamp_min(1e-30)),
                                  "lam*|g_render_proj| / |g_phys|": float(lam_r) * float(c_.norm()) / max(float(a_.norm()), 1e-30)}
                _rep["render_influence"] = _infl
                # finite differences through the SAME no-grad rollout the line search uses
                _leaf0 = [l.detach().clone() for l in leaves]
                _scales = [(float(cfg.dfc_clip) if cfg.dfc_clip > 0 else 0.02) if l is dFc else
                           ((float(sp0) if sp0 else 0.03) if (u is not None and l is u) else 1e-3) for l in leaves]
                def _set(vals):
                    with torch.no_grad():
                        for l, v in zip(leaves, vals):
                            l.copy_(v)
                def _eval():
                    with torch.no_grad():
                        st_, lv_, lk_, lr_, lpbr_, ex_ = eval_terms(dFc)
                        Lp_ = phys_core(lv_, lk_, ex_["dfc"], st_[0], st_[1], ex_["lk_run"],
                                        ex_["Fg"] if use_geom else None, ex_["lk_var"], _vT(ex_))
                        Ld_ = dt_term(st_[0])
                    return {"phys": float(Lp_), "render": float(lr_) if lr_ is not None else float("nan"),
                            "cleanup": float(Ld_) if Ld_ is not None else 0.0, "transport": float(lv_)}
                _set(_leaf0)
                _e0a, _e0b = _eval(), _eval()
                _rep["graph_vs_eval_at_leaf0"] = {"phys_graph": float(Lp_core.detach()), "phys_eval": _e0a["phys"],
                                                  "render_graph": float(lr.detach()), "render_eval": _e0a["render"],
                                                  "cleanup_graph": float(Ldt.detach()) if Ldt is not None else 0.0,
                                                  "cleanup_eval": _e0a["cleanup"]}
                _rep["replay_noise"] = {k_: abs(_e0a[k_] - _e0b[k_]) for k_ in _e0a}
                _gen = torch.Generator(device=dev).manual_seed(0)
                def _unit(ds):
                    return [d_ / d_.abs().max().clamp_min(1e-30) * sc_ for d_, sc_ in zip(ds, _scales)]
                _dirs = {"-g_phys": _unit([-g_ for g_ in _gp]), "-g_render": _unit([-g_ for g_ in _gr_true]),
                         "rand0": _unit([torch.randn(l.shape, device=dev, generator=_gen) for l in leaves]),
                         "rand1": _unit([torch.randn(l.shape, device=dev, generator=_gen) for l in leaves])}
                if _gd is not None:
                    _dirs["-g_cleanup"] = _unit([-g_ for g_ in _gd])
                _grads = {"phys": _gp, "render": _gr_true, "cleanup": _gd}
                _fd = {}
                for dn_, d_ in _dirs.items():
                    rows_ = []
                    for h_ in (0.1, 0.03, 0.01):
                        _set([l0 + h_ * dd for l0, dd in zip(_leaf0, d_)]); ep_ = _eval()
                        _set([l0 - h_ * dd for l0, dd in zip(_leaf0, d_)]); em_ = _eval()
                        row_ = {"h": h_}
                        for k_ in ("phys", "render", "cleanup"):
                            g_ = _grads[k_]
                            row_[k_] = {"fd": (ep_[k_] - em_[k_]) / (2 * h_),
                                        "autograd": _dot(g_, d_) if g_ is not None else 0.0}
                        rows_.append(row_)
                    _fd[dn_] = rows_
                _set(_leaf0)
                _rep["finite_difference"] = _fd
                # momentum budget of the rollout under four control settings
                _m = torch.as_tensor(np.asarray(m_np, np.float32) if not np.isscalar(m_np) else np.full(N, float(m_np), np.float32), device=dev)
                _M = float(_m.sum())
                def _budget(tag_, vals_):
                    _set(vals_)
                    with torch.no_grad():
                        eval_terms(dFc)
                        X_ = torch.stack([wp.to_torch(tr_eval.x[t_]).reshape(N, 3) for t_ in range(T + 1)])
                        V_ = torch.stack([wp.to_torch(tr_eval.v[t_]).reshape(N, 3) for t_ in range(T + 1)])
                        P_ = (_m[None, :, None] * V_).sum(1)                                   # (T+1, 3)
                        absP_ = (_m[None, :] * V_.norm(dim=2)).sum(1)                           # sum m|v|
                        C_ = (_m[None, :, None] * X_).sum(1) / _M                               # COM
                        R_ = X_ - C_[:, None, :]
                        L_ = (_m[None, :, None] * torch.cross(R_, V_, dim=2)).sum(1)
                        absL_ = (_m[None, :] * (R_.norm(dim=2) * V_.norm(dim=2))).sum(1)
                        dCom_ = (C_[-1] - C_[0]).norm()
                        # COM displacement explained by momentum (trapezoid) vs the actual one
                        dt_ = float(prm.dt)
                        mom_disp_ = dt_ * P_[1:].sum(0) / _M          # x_{t+1} = x_t + dt v_{t+1} (symplectic Euler)
                        vint_ = dt_ * V_[1:].sum(0)                     # the displacement the velocities account for
                        edit_ = (X_[-1] - X_[0]) - vint_               # position edits (layer projection, u, corrections)
                        lm_ = (torch.as_tensor(np.asarray(layer[0]) > 0.5, device=dev) if layer is not None
                               else torch.ones(N, dtype=torch.bool, device=dev))
                        stepC_ = (C_[1:] - C_[:-1]).norm(dim=1)
                    return {"|P|/sum m|v| max": float((P_.norm(dim=1) / absP_.clamp_min(1e-30))[1:].max()),
                            "|P|/sum m|v| end": float(P_[-1].norm() / absP_[-1].clamp_min(1e-30)),
                            "|L|/sum m|r||v| max": float((L_.norm(dim=1) / absL_.clamp_min(1e-30))[1:].max()),
                            "|L|/sum m|r||v| end": float(L_[-1].norm() / absL_[-1].clamp_min(1e-30)),
                            "sum m|v| end / max": float(absP_[-1] / absP_.max().clamp_min(1e-30)),
                            "COM displacement (wu)": float(dCom_),
                            "COM displacement explained by momentum (wu)": float(mom_disp_.norm()),
                            "COM displacement unexplained (wu)": float(((C_[-1] - C_[0]) - mom_disp_).norm()),
                            "max COM step (wu)": float(stepC_.max()),
                            "driven-phase end P/M (wu/s)": float(P_[min(cfg.T, T)].norm() / _M),
                            "rollout end P/M (wu/s)": float(P_[-1].norm() / _M), "sum m|v| max / M (wu/s)": float(absP_.max() / _M),
                            "layer |edit| median (wu)": float(edit_[lm_].norm(dim=1).median()), "layer |edit| p95 (wu)": float(torch.quantile(edit_[lm_].norm(dim=1), .95)),
                            "layer |v-displacement| median (wu)": float(vint_[lm_].norm(dim=1).median()), "layer |v-displacement| p95 (wu)": float(torch.quantile(vint_[lm_].norm(dim=1), .95)),
                            "interior |edit| p95 (wu)": float(torch.quantile(edit_[~lm_].norm(dim=1), .95)) if bool((~lm_).any()) else 0.0,
                            "interior |v-displacement| median (wu)": float(vint_[~lm_].norm(dim=1).median()) if bool((~lm_).any()) else 0.0,
                            "layer share": float(lm_.float().mean())}
                _zeros = [torch.zeros_like(l) for l in leaves]
                _dfc_only = [l0 if l is dFc else torch.zeros_like(l) for l, l0 in zip(leaves, _leaf0)]
                _u_only = [torch.zeros_like(l) if l is dFc else l0 for l, l0 in zip(leaves, _leaf0)]
                _step_leaf = [l0 - float(cfg.dfc_clip if cfg.dfc_clip > 0 else 0.02) * dd / dd.abs().max().clamp_min(1e-30) * (1.0 if l is dFc else 0.0)
                              for l, l0, dd in zip(leaves, _leaf0, _gp)]
                _rep["momentum"] = {"current controls": _budget("cur", _leaf0), "no control (dFc=0, u=0)": _budget("zero", _zeros),
                                    "dFc only": _budget("dfc", _dfc_only), "u only": _budget("u", _u_only),
                                    "dFc = clip-scaled -g_phys, u = 0": _budget("step", [st if l is dFc else torch.zeros_like(l) for l, st in zip(leaves, _step_leaf)]),
                                    "u = spacing-scaled -g_render, dFc = 0": _budget("ustep", [(-(g_ / g_.abs().max().clamp_min(1e-30)) * float(sp0 if sp0 else 0.03)) if (u is not None and l is u) else torch.zeros_like(l) for l, g_ in zip(leaves, _gr_true)]),
                                    "both steps": _budget("both", [st if l is dFc else ((-(g_ / g_.abs().max().clamp_min(1e-30)) * float(sp0 if sp0 else 0.03)) if (u is not None and l is u) else torch.zeros_like(l)) for l, st, g_ in zip(leaves, _step_leaf, _gr_true)])}
                _set(_leaf0)
                _rep["mass_total"] = _M; _rep["dt"] = float(prm.dt); _rep["drag"] = float(prm.drag)
                with open(os.environ["MJ_GRADCHECK"], "w") as f_:
                    _json.dump(_rep, f_, indent=1)
                print(f"[gradcheck] report written to {os.environ['MJ_GRADCHECK']}", flush=True)
                os._exit(0)
'''
s = s.replace(anchor, hook + anchor); n += 1
open(p, "w").write(s)
print(f"patched {p}: {n} edits")
