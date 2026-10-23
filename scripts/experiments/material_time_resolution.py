"""External temporal refinement: same window time and coarse control variables."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import patch


def integration_timing(dt, T, substeps):
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError('substeps must be a positive integer')
    return dt/substeps, T*substeps


def expand_control(leaf, substeps):
    import torch
    driven = leaf.repeat_interleave(substeps, dim=0)/substeps
    return torch.cat((driven, torch.zeros_like(driven)), dim=0)


@contextmanager
def refine_time(substeps=2):
    """Patch only the external entry's timing and control discretisation, not the solver."""
    import torch
    from scripts import pipeline_run as P
    from physmorph.pipeline.window.setup import Window
    from physmorph.pipeline.window.solve import WindowOptimizer
    integration_timing(1., 1, substeps)
    info = dict(substeps=substeps, control_mapping='hold each coarse dFc for substeps, each divided by substeps',
        safety='material_state_safety remains independently enabled by the caller')
    original_prepare, original_config = P.prepare, P.run_config
    original_init = WindowOptimizer.__init__

    def prepare(*args, **kwargs):
        prep = original_prepare(*args, **kwargs)
        info.update(base_dt=prep.prm.dt, refined_dt=prep.prm.dt/substeps,
                    volume=prep.v_src, particles=len(prep.src))
        return replace(prep, prm=replace(prep.prm, dt=prep.prm.dt/substeps))

    def config(args, cfg0, prep):
        cfg = original_config(args, cfg0, prep)
        dt, T = integration_timing(info['base_dt'], cfg.T, substeps)
        assert dt == prep.prm.dt
        info.update(base_T=cfg.T, integration_T=T, total_window_time=2*T*dt,
                    optimized_dfc_shape=[cfg.T, len(prep.src), 3, 3])
        rho = (cfg.mass_ref_n if cfg.mass_ref_n > 0 else len(prep.src))/prep.v_src
        lam = cfg.young*cfg.poisson/((1+cfg.poisson)*(1-2*cfg.poisson))
        mu = cfg.young/(2*(1+cfg.poisson))
        c = ((lam+2*mu)/rho)**.5
        info.update(density=rho, cfl_before=c*info['base_dt']/prep.prm.dx, cfl_after=c*dt/prep.prm.dx)
        return replace(cfg, T=T)

    def init(opt, *args, **kwargs):
        original_init(opt, *args, **kwargs)
        coarse_T = opt.cfg.T//substeps
        opt.dFc = opt.dFc.new_zeros((coarse_T,opt.win.N,3,3), requires_grad=True)
        opt.leaves = [opt.dFc,opt.u]
        opt.mom = [torch.zeros_like(p) for p in opt.leaves]
        opt.vel = [torch.zeros_like(p) for p in opt.leaves]

    def expand(win, leaf):
        if leaf.shape[0]*substeps != win.cfg.T:
            raise ValueError('Control horizon does not match refined integration horizon')
        return expand_control(leaf, substeps)

    with ExitStack() as stack:
        stack.enter_context(patch.object(P,'prepare',prepare))
        stack.enter_context(patch.object(P,'run_config',config))
        stack.enter_context(patch.object(WindowOptimizer,'__init__',init))
        stack.enter_context(patch.object(Window,'expand',expand))
        yield info
