"""Diagnostic linear descent over a trust ball and up to eight affine halfspaces."""
import itertools
import math

import torch


def geometric_running(positions,x0,dt,mask):
    """All saved physical steps, including the initial position -> first step."""
    if not math.isfinite(dt) or dt<=0 or not len(positions) or not bool(mask.any()):
        raise ValueError('Invalid running-motion domain')
    path = torch.cat((x0[None],positions))
    return ((path[1:,mask]-path[:-1,mask])/dt).square().sum(-1).mean()


def _linear_arrays(step,constraints,bounds):
    if (bounds.ndim!=1 or constraints.shape!=(len(bounds),*step.shape) or
            constraints.device!=step.device or bounds.device!=step.device or not step.numel()):
        raise ValueError('Invalid affine array layout')
    arrays = step.detach().flatten().double(),constraints.detach().reshape(len(bounds),step.numel()).double(),bounds.detach().double()
    if not all(bool(torch.isfinite(x).all()) for x in arrays):
        raise ValueError('Nonfinite affine problem')
    return arrays


@torch.no_grad()
def affine_ball_step(gradient, constraints, bounds, radius):
    """Return a float64 step; callers must recheck after coefficient projection.

    Minimize gradient.dot(step), with G @ step <= b and ||step|| <= radius.
    Enumerating active planes is exact in real arithmetic for up to eight rows.
    A failed numerical active-set search is not a global nonlinear certificate.
    """
    shape = gradient.shape
    g,G,b = _linear_arrays(gradient,constraints,bounds)
    if len(b)>8 or not math.isfinite(radius) or radius<0:
        raise ValueError('Invalid affine trust-ball problem')
    norms = G.norm(dim=1)
    if not bool(torch.isfinite(norms).all() & torch.isfinite(g.norm())):
        raise ValueError('Nonfinite affine norms')
    zero = norms==0
    if bool((zero & (b<0)).any()):
        return None,dict(status='zero_gradient_cannot_restore')
    retained_rows = torch.nonzero(~zero).flatten()
    G,b = G[~zero]/norms[~zero,None],b[~zero]/norms[~zero]
    eps = torch.finfo(g.dtype).eps
    feasible = []
    for size in range(len(b)+1):
        for active in itertools.combinations(range(len(b)),size):
            p = torch.zeros_like(g)
            descent = -g
            if active:
                A = G[list(active)]; rhs = b[list(active)]
                U,s,Vh = torch.linalg.svd(A,full_matrices=False)
                keep = s>max(A.shape)*eps*s.max()
                if len(s)==A.shape[0] and bool(keep.all()):
                    # Avoid forming the squared-condition normal equations.
                    Q,R = torch.linalg.qr(A.T,mode='reduced')
                    p = Q@torch.linalg.solve_triangular(R.T,rhs[:,None],upper=False).flatten()
                    basis = Q.T
                else:
                    basis = Vh[keep]
                    p = basis.T@((U[:,keep].T@rhs)/s[keep])
                descent = descent-basis.T@(basis@descent)
                tol = 64*eps*(1+radius+float(rhs.norm())+float(p.norm()))
                if bool(((A@p-rhs).abs()>tol).any()):
                    continue
            tol = 64*eps*(1+radius+float(b.norm())+float(p.norm()))
            remaining = radius*radius-float(p.square().sum())
            if remaining < -tol*(1+radius):
                continue
            norm = descent.norm()
            candidate = p
            if float(norm)>64*eps*float(g.norm()):
                candidate = p+math.sqrt(max(0.,remaining))*descent/norm
            if (float(candidate.norm())>radius+tol or
                    bool((G@candidate>b+tol).any())):
                continue
            feasible.append((candidate,active))
    if not feasible:
        return None,dict(status='no_feasible_active_set')
    scores = torch.stack([g@p for p,_ in feasible])
    chosen = int(scores.argmin())
    step,active = feasible[chosen]
    return step.reshape(shape),dict(status='linear_candidate',active=list(active),
        active_input_rows=[int(retained_rows[i]) for i in active],
        norm=float(step.norm()),radius=radius,linear_objective=float(scores[chosen]),
        feasible_active_sets=len(feasible))


@torch.no_grad()
def projected_affine_check(step,constraints,bounds,input_dtype):
    """Account only for input-precision arithmetic, never relax nonlinear gates."""
    delta,G,b = _linear_arrays(step,constraints,bounds)
    residual = G@delta-b
    tolerance = 32*torch.finfo(input_dtype).eps*(G.norm(dim=1)*delta.norm()+b.abs())
    if not bool(torch.isfinite(residual).all() & torch.isfinite(tolerance).all()):
        raise ValueError('Nonfinite projected affine check')
    return dict(passed=bool((residual<=tolerance).all()),
                residual=[float(v) for v in residual],tolerance=[float(v) for v in tolerance])


@torch.no_grad()
def observed_remainder(actual,origin,constraints,projected_step):
    """Actual minus the frozen affine model; not a curvature/noise certificate."""
    delta,G,f0 = _linear_arrays(projected_step,constraints,origin)
    if actual.shape!=origin.shape or actual.device!=origin.device:
        raise ValueError('Invalid observed-data layout')
    remainder = actual.detach().double()-f0-G@delta
    if not bool(torch.isfinite(actual).all() & torch.isfinite(remainder).all()):
        raise ValueError('Nonfinite observed remainder')
    return remainder
