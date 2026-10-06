# D116 diagnostic block (not part of the package): inserted after `rec = _record(...)` in run/runner.py of a scratch
# copy (with `import os`), active with D116_FREE=1; records the zero-control rollout of every committed window.
        if os.environ.get("D116_FREE"):             # D116 diagnostic (scratch copy only): the window's free moves
            import dataclasses as _dc
            from ..window.objective import Objective as _Obj
            from ..window.rollout import eval_terms as _ev
            from ..window.setup import Window as _Win
            from ...render.knn_gpu import knn_self_torch as _knn
            with torch.no_grad():
                _d, _nb = _knn(x_start, 17)
                _off = x_start[_nb[:, 1:]].mean(1) - x_start
                _outer = _off.norm(dim=1) > 0.25 * _d[:, 1:].median(1).values
                _n = -_off[_outer] / _off[_outer].norm(dim=1, keepdim=True)
                _p = float(_d[:, 1].median())
                _act = ((x - x_start)[_outer] * _n).sum(1) / _p
            _st = StartState(x=x_start, Fp=Fp_pre, F=rollback["F"], v=rollback["v"], C=rollback["C"])
            _bonds = (coh_nbr, bond_rest, frag.float())
            _out = {"act_n": float(_act.square().mean().sqrt())}
            for _tag, _c in (("free", cfg), ("free_nospace", _dc.replace(cfg, min_spacing=0.0)),
                             ("free_norelax", _dc.replace(cfg, baseline="xu_diag"))):
                _w = _Win(_st, prm, _c, tgt, vol0, _bonds)
                with torch.no_grad():
                    _e = _ev(_w, _Obj(_w), torch.zeros(_w.Tc, _w.N, 3, 3, device=x.device), torch.zeros(_w.N, device=x.device))
                    _fr = ((_e.xT - x_start)[_outer] * _n).sum(1) / _p
                _out[_tag + "_n"] = float(_fr.square().mean().sqrt())
                if _tag == "free":
                    _out["corr_act_free"] = float(torch.corrcoef(torch.stack((_act, _fr)))[0, 1])
                    _out["push_n"] = float((_act - _fr).square().mean().sqrt())
                del _w
            rec.update({"d116_" + k: v_ for k, v_ in _out.items()})
            log(f"[d116] anim {a + 1}: " + " ".join(f"{k}={v_:.4g}" for k, v_ in _out.items()))
