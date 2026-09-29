"""World-space observations of supplied Gaussian covariances, never a renderer.

No F is inferred, no camera projection is performed, and no input is changed.
All per-particle math stays on the input device. Only one reduced scalar packet
crosses to the host; caller-defined masks do not cause dynamic compaction.
"""
from collections.abc import Mapping

import torch


@torch.no_grad()
def summarize_world_footprints(covariance, normals, *, support, opacity, populations=None):
    """Observe normal/tangent standard deviations and their covariance coupling.

    Inputs are float32/float64 tensors on one device: covariance (N,3,3), unit
    normals (N,3), support/opacity (N,) in [0,1]. Named population masks are bool
    (N,) tensors on that device. Invalid numerical rows are counted and excluded
    explicitly, not repaired. Empty summaries are None. No statistic certifies
    screen coverage, physical supply, watertightness or particle rest.
    """
    if not torch.is_tensor(covariance) or covariance.ndim != 3 or covariance.shape[1:] != (3, 3):
        raise ValueError('covariance must have shape (N,3,3)')
    n, device = len(covariance), covariance.device
    for name, value, shape in (('covariance', covariance, (n, 3, 3)), ('normals', normals, (n, 3)),
                               ('support', support, (n,)), ('opacity', opacity, (n,))):
        if (not torch.is_tensor(value) or value.shape != shape or value.device != device
                or value.dtype not in (torch.float32, torch.float64)):
            raise ValueError(f'{name} requires the declared shape/device and float32 or float64')
    if populations is not None and not isinstance(populations, Mapping):
        raise ValueError('populations must be a mapping of names to bool masks')
    masks = {'all': torch.ones(n, dtype=torch.bool, device=device),
             'positive_support_and_opacity': (support > 0) & (opacity > 0),
             'zero_opacity': opacity == 0}
    for name, mask in (populations or {}).items():
        if not isinstance(name, str) or not name or name in masks:
            raise ValueError('Population names must be nonempty and must not replace built-in populations')
        if not torch.is_tensor(mask) or mask.shape != (n,) or mask.dtype != torch.bool or mask.device != device:
            raise ValueError('Population masks must be bool (N,) tensors on the input device')
        masks[name] = mask

    eps64 = 64*torch.finfo(torch.float64).eps
    normal_tol, symmetry_tol = 64*torch.finfo(normals.dtype).eps, 64*torch.finfo(covariance.dtype).eps
    c, normal, supp, alpha = (value.to(torch.float64) for value in (covariance, normals, support, opacity))
    finite_c = torch.isfinite(c).all(dim=(1, 2))
    finite_n = torch.isfinite(normal).all(dim=1)
    c = torch.where(torch.isfinite(c), c, 0.)
    normal = torch.where(torch.isfinite(normal), normal, 0.)
    scale = c.abs().amax(dim=(1, 2))
    symmetric = (c-c.transpose(1, 2)).abs().amax(dim=(1, 2)) <= symmetry_tol*scale
    # The symmetric part defines the quadratic form; asymmetry beyond the stated
    # input-precision allowance is invalid. Normalize only for stable diagnostics.
    c = (.5*c+.5*c.transpose(1, 2))/torch.where(scale > 0, scale, 1.)[:, None, None]
    norm2 = normal.square().sum(1)
    unit = (norm2-1.).abs() <= normal_tol
    normal = normal/torch.sqrt(torch.where(norm2 > 0, norm2, 1.))[:, None]

    # All principal minors characterize PSD for a symmetric 3x3 matrix. Their
    # bounds use sums of absolute products in float64, not a geometric cutoff.
    a, d, f = c[:, 0, 0], c[:, 1, 1], c[:, 2, 2]
    b, q, e = c[:, 0, 1], c[:, 0, 2], c[:, 1, 2]
    minors = [(a*d, b*b), (a*f, q*q), (d*f, e*e)]
    psd = (a >= 0) & (d >= 0) & (f >= 0)
    for positive, negative in minors:
        psd &= positive-negative >= -eps64*(positive.abs()+negative.abs())
    determinant_terms = (a*d*f, 2*b*q*e, -a*e*e, -d*q*q, -f*b*b)
    psd &= sum(determinant_terms) >= -eps64*sum(value.abs() for value in determinant_terms)

    # Select a well-conditioned reference axis; no particle selection/copy to CPU.
    axis = torch.zeros_like(normal).scatter_(1, normal.abs().argmin(1)[:, None], 1.)
    tangent1 = torch.linalg.cross(normal, axis, dim=1)
    tangent1 /= torch.linalg.vector_norm(tangent1, dim=1).clamp_min(torch.finfo(torch.float64).tiny)[:, None]
    tangent2 = torch.linalg.cross(normal, tangent1, dim=1)
    def quadratic(left, right):
        return (left*torch.einsum('nij,nj->ni', c, right)).sum(1)
    nn = quadratic(normal, normal)
    tt1, tt2, cross = quadratic(tangent1, tangent1), quadratic(tangent2, tangent2), quadratic(tangent1, tangent2)
    spread = torch.hypot(tt1-tt2, 2*cross)
    major = .5*(tt1+tt2+spread)
    # Product/major avoids cancellation in the smaller eigenvalue of a thin disc.
    divisor = torch.where(major > 0, major, 1.)
    minor = (tt1/divisor)*tt2-(cross/divisor)*cross
    variance_tol = eps64*c.abs().sum(dim=(1, 2))
    variance_ok = (nn >= -variance_tol) & (minor >= -variance_tol) & (major >= -variance_tol)
    support_ok = torch.isfinite(supp) & (supp >= 0) & (supp <= 1)
    opacity_ok = torch.isfinite(alpha) & (alpha >= 0) & (alpha <= 1)
    valid = finite_c & finite_n & symmetric & unit & psd & variance_ok & support_ok & opacity_ok
    roundoff = valid & ((nn < 0) | (minor < 0) | (major < 0))
    root_scale = torch.sqrt(scale)
    normal_std = torch.sqrt(torch.where(valid, nn.clamp_min(0), 0.))*root_scale
    minor_std = torch.sqrt(torch.where(valid, minor.clamp_min(0), 0.))*root_scale
    major_std = torch.sqrt(torch.where(valid, major.clamp_min(0), 0.))*root_scale
    coupling = torch.hypot(quadratic(tangent1, normal), quadratic(tangent2, normal))*scale
    observable_ok = torch.isfinite(torch.stack((normal_std, minor_std, major_std, coupling), 1)).all(1)
    valid &= observable_ok
    roundoff &= valid
    anisotropy = major_std/torch.where(minor_std > 0, minor_std, 1.)

    packet = []
    def scalar(value):
        packet.append(value.to(torch.float64))
        return len(packet)-1
    def summary(value, mask):
        count = mask.sum()
        clean = torch.where(mask, value, 0.)
        # Appended sentinels make min/max well-defined even for N=0. The host
        # emits None for an empty summary; infinities never enter its JSON values.
        low = torch.cat((torch.where(mask, value, torch.inf), value.new_tensor([torch.inf]))).min()
        high = torch.cat((torch.where(mask, value, -torch.inf), value.new_tensor([-torch.inf]))).max()
        # All observables are nonnegative. Scaling by their maximum avoids
        # squaring/summing overflow without excluding a finite, very thin ellipsoid.
        divisor = torch.where(high > 0, high, 1.)
        scaled = clean/divisor
        return [scalar(v) for v in (count, scaled.sum()/count.clamp_min(1)*divisor,
                torch.sqrt(scaled.square().sum()/count.clamp_min(1))*divisor, low, high)]
    failures = {'nonfinite_covariance': ~finite_c, 'nonfinite_normal': ~finite_n,
                'asymmetric_covariance': ~symmetric, 'nonunit_normal': ~unit,
                'non_psd_covariance': ~psd, 'negative_observed_variance': ~variance_ok,
                'invalid_support': ~support_ok, 'invalid_opacity': ~opacity_ok,
                'nonfinite_observable': ~observable_ok}
    validation = {name: scalar(mask.sum()) for name, mask in failures.items()}
    invalid_index, clamp_index = scalar((~valid).sum()), scalar(roundoff.sum())
    populations_packet = {}
    for name, mask in masks.items():
        selected = mask & valid
        counts = {key: scalar(value.sum()) for key, value in {
            'selected_count': mask, 'valid_count': selected, 'invalid_count': mask & ~valid,
            'zero_support_count': selected & (supp == 0), 'zero_opacity_count': selected & (alpha == 0),
            'anisotropy_unbounded_count': selected & (minor_std == 0) & (major_std > 0),
            'anisotropy_undefined_count': selected & (minor_std == 0) & (major_std == 0),
            'anisotropy_overflow_count': selected & (minor_std > 0) & ~torch.isfinite(anisotropy)}.items()}
        observations = {key: summary(value, selected) for key, value in {
            'normal_std_wu': normal_std, 'tangent_minor_std_wu': minor_std,
            'tangent_major_std_wu': major_std, 'normal_tangent_covariance_wu2': coupling,
            'support': supp, 'opacity': alpha}.items()}
        observations['tangent_anisotropy'] = summary(anisotropy, selected & (minor_std > 0) & torch.isfinite(anisotropy))
        populations_packet[name] = counts, observations
    values = torch.stack(packet).cpu().tolist()
    report = dict(schema='world_gaussian_footprint_v1', input_count=n,
        valid=values[invalid_index] == 0, invalid_count=int(values[invalid_index]),
        validation_counts={key: int(values[index]) for key, index in validation.items()},
        roundoff_clamped_rows=int(values[clamp_index]),
        numerics=dict(arithmetic='float64', normal_squared_norm_tolerance=normal_tol,
            relative_symmetry_tolerance=symmetry_tol, arithmetic_error_factor=eps64,
            psd='all principal minors, normalized by maximum covariance entry; '
                '64 float64 eps times each absolute-product sum',
            roundoff='Only tolerance-sized negative observed variances are clamped for square roots; '
                     'invalid rows are excluded, input covariance is never repaired',
            reductions='Nonnegative means/RMS are normalized by the selected maximum to avoid overflow'),
        definitions=dict(normal='sqrt(n^T Sigma n), along the supplied unit-normal direction',
            tangent='principal standard deviations of Sigma restricted to the plane perpendicular to n',
            coupling='norm of tangent components of Sigma n, covariance units wu^2; '
                     'nonzero coupling identifies an ellipsoid tilted relative to the supplied normal',
            basis='supplied normal is normalized to form an orthonormal diagnostic frame after unit validation',
            populations='all includes zero-opacity/support rows; positive_support_and_opacity is a proxy, '
                        'not camera visibility; user masks are supplied, not inferred',
            validity='All statistics exclude rows invalid in any required input; validation categories may overlap',
            scope='World-space diagnostic only; not raster-exact, not projected pixel radii, '
                  'not a physical supply/hole/rest metric or a footprint intervention'), populations={})
    for name, (counts, observations) in populations_packet.items():
        row = {key: int(values[index]) for key, index in counts.items()}
        for key, indices in observations.items():
            count, mean, rms, low, high = (values[index] for index in indices)
            row[key] = (dict(count=int(count), mean=mean, rms=rms, minimum=low, maximum=high)
                        if count else None)
        report['populations'][name] = row
    return report
