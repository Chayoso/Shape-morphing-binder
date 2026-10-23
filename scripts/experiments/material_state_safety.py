"""External-only state localization and safe native candidate rejection."""
from contextlib import ExitStack, contextmanager
import copy
from unittest.mock import patch


class LocalizedInvalidState(Exception):
    pass


class InvalidStart(Exception):
    pass


def trajectory_check(opt, details=False, trajectory=None, controls=None):
    import torch
    import warp as wp
    from physmorph.pipeline.window.setup import domain_bounds
    from physmorph.pipeline.window.rollout import _trajectory_min_det
    win = opt.win; tr = win.tr if trajectory is None else trajectory
    x = torch.stack([wp.to_torch(a) for a in tr.x])
    v = torch.stack([wp.to_torch(a) for a in tr.v])
    F = torch.stack([wp.to_torch(a).reshape(win.N, 3, 3) for a in tr.F])
    lo, hi = domain_bounds(win.prm, x.device)
    finite = [torch.isfinite(a).flatten(2).all(2) for a in (x, v, F)]
    domain = ((x >= lo) & (x <= hi)).all(2) & finite[0]
    bad_finite = ~(finite[0] & finite[1] & finite[2])
    reason = 'nonfinite' if bool(bad_finite.any()) else 'domain' if not bool(domain.all()) else None
    # The existing whole-trajectory orientation check, moved before losses.
    dc = win.dc_buf if controls is None else controls
    view = win if trajectory is None else copy.copy(win)
    if trajectory is not None: view.tr = tr
    jt = _trajectory_min_det(view, dc) if reason != 'nonfinite' else float('nan')
    if reason is None and not jt > 1e-4: reason = 'jt'
    row = dict(reason=reason, jt=jt, nonfinite_counts=[int((~a).sum()) for a in finite],
               outside_positions=int((~domain).sum()))
    if details and reason is not None:
        det = torch.linalg.det(F)
        effective_det = torch.linalg.det(F[:-1] + dc)
        row['nonfinite_kinds']=[dict(nan=int(torch.isnan(a).sum()),infinite=int(torch.isinf(a).sum())) for a in (x,v,F)]
        def first(mask):
            where = torch.nonzero(mask, as_tuple=False)
            return where[0].tolist() if len(where) else None
        row.update(first_nonfinite=first(bad_finite), first_outside=first(~domain),
                   first_bad_F_det=first(~(det > 1e-4)),
                   first_bad_effective_det=first(~(effective_det > 1e-4)))
        at = row['first_nonfinite'] or row['first_outside'] or row['first_bad_F_det'] or row['first_bad_effective_det']
        if at is not None:
            t, p = at; row['particle'] = p; row['first_bad_state_timestep'] = t
            row['nearby_states'] = [dict(timestep=k, position=x[k,p].tolist(), velocity=v[k,p].tolist(),
                F=F[k,p].tolist(), F_det=float(det[k,p]),
                effective_det=float(effective_det[k,p]) if k < win.T else None,
                control=dc[k,p].tolist() if k < win.T else None) for k in range(max(0,t-1),min(win.T,t+1)+1)]
        row['timesteps'] = [dict(timestep=t, bad_x=int((~finite[0][t]).sum()),
            bad_v=int((~finite[1][t]).sum()), bad_F=int((~finite[2][t]).sum()),
            outside=int((~domain[t]).sum()), F_det_min=float(det[t].amin()),
            x_absmax=float(x[t].abs().amax()), v_absmax=float(v[t].abs().amax()),
            F_absmax=float(F[t].abs().amax())) for t in range(win.T+1)]
    return row


@contextmanager
def state_context(emit, diagnostic=False):
    import torch
    from physmorph.mpm.traj import Trajectory
    from physmorph.pipeline.window import solve as S, rollout as R
    from physmorph.pipeline.window.objective import Objective
    old_init, old_run = S.WindowOptimizer.__init__, S.WindowOptimizer.run
    old_line, old_warm, old_replay = S.WindowOptimizer.line_search, S.WindowOptimizer.warm_start, S.WindowOptimizer.replay_noise
    old_commit=S.WindowOptimizer.commit
    old_eval, old_evaluate, old_reason, old_scalar = R.eval_terms, R._evaluate, R.state_reason, Objective.scalar
    old_tr_run = Trajectory.run
    ctx = dict(opt=None, window=0, phase='startup', calls=0, trial=0)
    stats = dict(evaluations=0, invalid=0, reasons={}, invalid_trial_scores=0,
                 recovered_lines=0, exhausted_rollbacks=0, warm_zero_fallbacks=0, aborted_windows=0,
                 graph_evaluations=0, invalid_graphs=0)
    seen = set(); first = set()
    def meta():
        opt = ctx['opt']
        phase=ctx['phase']
        if phase=='warm_start':phase='warm_start_zero' if ctx.get('warm_call',0)==1 else 'warm_start_candidate'
        return dict(window=ctx['window'], iteration=getattr(opt,'_it',None), phase=phase, evaluation=ctx['calls'], trial=ctx['trial'])
    def note(row):
        stats['invalid'] += 1; reason=row['reason']; stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        key=(ctx['window'],getattr(ctx['opt'],'_it',None),ctx['phase'],reason)
        if diagnostic or key not in seen:
            emit(dict(kind='invalid_state',**meta(),**row)); seen.add(key)
    def init(opt,*args,**kwargs):
        old_init(opt,*args,**kwargs); ctx.update(opt=opt,window=ctx['window']+1,phase='startup',calls=0,trial=0)
    def line(opt,*args,**kwargs):
        before=ctx['phase']; ctx.update(phase='trial',trial=0); invalid=stats['invalid']
        snapshots=[p.detach().clone() for p in opt.leaves+opt.mom+opt.vel] if not diagnostic else []; adam_t=opt.adam_t
        try: result=old_line(opt,*args,**kwargs)
        finally: ctx['phase']=before
        if not diagnostic and stats['invalid']>invalid:
            if result is None:
                same=all(torch.equal(a,b) for a,b in zip(opt.leaves+opt.mom+opt.vel,snapshots)) and opt.adam_t==adam_t
                if not same: raise RuntimeError('External safety: native exhausted-search rollback mismatch')
                stats['exhausted_rollbacks']+=1
            else: stats['recovered_lines']+=1
        return result
    def warm(opt,*args,**kwargs):
        before=ctx['phase'];ctx.update(phase='warm_start',trial=0,warm_call=0); bad=stats['invalid']
        try: result=old_warm(opt,*args,**kwargs)
        finally:ctx['phase']=before
        if not diagnostic and stats['invalid']>bad and not bool(opt.dFc.detach().count_nonzero()):
            stats['warm_zero_fallbacks']+=1
        return result
    def replay(opt,*args,**kwargs):
        before=ctx['phase'];ctx['phase']='replay'
        try:result=old_replay(opt,*args,**kwargs)
        finally:ctx['phase']=before
        if not diagnostic and opt.tele.get('start_ok')==0:raise InvalidStart('Both retained start/replay state invalid')
        return result
    def commit(opt,*args,**kwargs):
        before=ctx['phase'];ctx['phase']='commit'
        try:return old_commit(opt,*args,**kwargs)
        finally:ctx['phase']=before
    def run(opt,*args,**kwargs):
        balance=copy.copy(opt.balancer.__dict__) if hasattr(opt,'balancer') else None
        hint=opt.tgt.settled_step
        try:return old_run(opt,*args,**kwargs)
        except InvalidStart as error:
            if diagnostic:raise
            stats['aborted_windows']+=1
            with torch.no_grad():
                for t in opt.leaves+opt.mom+opt.vel:t.zero_()
            opt.adam_t=0;opt.tgt.settled_step=hint
            if balance is not None:opt.balancer.__dict__.clear();opt.balancer.__dict__.update(balance)
            # A graph can abort before transport consumes its same-point flag.
            # No endpoint was committed: the next rollout must solve afresh.
            if opt.tgt.grid_ot is not None:
                opt.tgt.grid_ot.repeat=False;opt.tgt.grid_ot.warm=False;opt.tgt.grid_ot._solved=None
            opt.win._adj=None
            opt.tele['null_reason']='external_invalid_start'
            emit(dict(kind='window_abort',**meta(),reason=str(error),transport_cache_invalidated=True,
                      previous_controls_restored_by_runner=True))
            return S.WindowResult(None,[],dict(opt.tele,accepted=0,rejected=0,grad_converged=False,
                replay_rel=0.,dfc=opt.dFc.detach().clone(),invalid_selection=True,null_reason='external_invalid_start'))
    def inspect(opt):
        stats['evaluations']+=1;ctx['calls']+=1
        if ctx['phase']=='trial':ctx['trial']+=1
        if ctx['phase']=='warm_start':ctx['warm_call']+=1
        row=trajectory_check(opt,details=False)
        if row['reason'] is not None:
            key=row['reason']
            if diagnostic or key not in first:
                row=trajectory_check(opt,details=True);first.add(key)
            note(row)
        return row
    def tr_run(tr,*args,**kwargs):
        result=old_tr_run(tr,*args,**kwargs);opt=ctx['opt']
        if opt is not None and tr is opt.win.tr:
            row=inspect(opt)
            if row['reason']=='nonfinite':
                emit(dict(kind='localization_complete',**meta(),first_nonfinite=row.get('first_nonfinite')))
                raise LocalizedInvalidState('First nonfinite native candidate localized before losses')
        return result
    def invalid_eval(win,xT,FT,vT,dc,V,row):
        bad=xT.new_full((),float('inf'))
        e=R.Eval(xT,FT,vT,bad,bad,bad,bad,bad,dc,V,bad,bad,lstab=bad,lk_drv=bad,lk_rel=bad,
                 jt=row['jt'],in_domain=row['reason']!='domain')
        e.external_invalid_reason=row['reason'];return e
    def evaluate(win,obj,leaf,u):
        import warp as wp
        with torch.no_grad():
            dc=win.load(leaf,u);win.tr.run();row=inspect(ctx['opt'])
            tr=win.tr;xT=wp.to_torch(tr.x[win.T]).clone();FT=wp.to_torch(tr.F[win.T]).reshape(win.N,9).clone()
            vT=wp.to_torch(tr.v[win.T]).clone();V=torch.stack([wp.to_torch(t) for t in tr.v[1:]])
            if row['reason'] is not None:return invalid_eval(win,xT,FT,vT,dc,V,row)
            e=old_evaluate(obj,xT,FT,vT,dc,V);e.jt=row['jt'];e.in_domain=True;return e
    def graph_evaluate(obj,xT,FT,vT,dc,V,**kw):
        opt=ctx['opt'];adj=opt.win._adj;stats['graph_evaluations']+=1
        with torch.no_grad():row=trajectory_check(opt,trajectory=adj.traj,controls=adj.dc)
        if row['reason'] is not None:
            stats['invalid_graphs']+=1
            with torch.no_grad():row=trajectory_check(opt,details=True,trajectory=adj.traj,controls=adj.dc)
            emit(dict(kind='invalid_graph_state',**meta(),**row))
            raise InvalidStart('Invalid graph trajectory before loss/adjoint: '+row['reason'])
        return old_evaluate(obj,xT,FT,vT,dc,V,**kw)
    def reason(e):return getattr(e,'external_invalid_reason',None) or old_reason(e)
    def scalar(obj,e,lam):
        if getattr(e,'external_invalid_reason',None) is not None:
            if ctx['phase']=='trial':stats['invalid_trial_scores']+=1
            return float('inf')
        return old_scalar(obj,e,lam)
    try:
        with ExitStack() as stack:
            for cls,name,value in ((S.WindowOptimizer,'__init__',init),(S.WindowOptimizer,'line_search',line),
                    (S.WindowOptimizer,'warm_start',warm),(S.WindowOptimizer,'replay_noise',replay),
                    (S.WindowOptimizer,'commit',commit),(S.WindowOptimizer,'run',run)):
                stack.enter_context(patch.object(cls,name,value))
            if diagnostic:stack.enter_context(patch.object(Trajectory,'run',tr_run))
            else:
                for cls,name,value in ((R,'eval_terms',evaluate),(S,'eval_terms',evaluate),(R,'_evaluate',graph_evaluate),
                       (R,'state_reason',reason),(S,'state_reason',reason),(Objective,'scalar',scalar)):
                    stack.enter_context(patch.object(cls,name,value))
            yield stats
    finally:emit(dict(kind='safety_summary',stats=stats,diagnostic=diagnostic))
