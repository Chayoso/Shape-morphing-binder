"""Small independent FP64 reference for a first-step moving pin collider.

Only pin anchors vary. Actual free-particle P2G mass/momentum are fixed inputs:
slip pins are excluded from P2G, so their anchor derivative cannot enter there.
Those inputs already contain stress, drag and any fixed APIC/viscosity state.
This covers pin-mass raster -> grid projection -> velocity G2P, not a whole
FP64 MPM step, Fp assimilation, or the derivative of admission/preparation.
"""
import torch


def first_step_pin_velocity(x0, pins, particle_mass, grid_mass, grid_momentum, prm):
    """All tensor inputs stay on their device; return FP64 values and branches."""
    x = x0.to(torch.float64)
    pins = pins.bool()
    shape = (prm.nx, prm.ny, prm.nz)
    if len(x) > 64 or prm.ngrid > 65536:
        raise ValueError('Dense independent reference is only for small test fixtures')
    indices = torch.stack(torch.meshgrid(*(torch.arange(size, device=x.device)
                                          for size in shape), indexing='ij'), dim=-1).reshape(-1, 3)
    nodes = indices.to(torch.float64)*prm.dx+x.new_tensor(prm.grid_min)
    distance = ((nodes[None]-x[:, None])/prm.dx).abs()
    # Cardinal cubic B-spline, evaluated independently in FP64 without calling
    # any Warp interpolation helper or using its rounded particle weights.
    one_d = torch.where(distance < 1., (4.-6.*distance.square()+3.*distance**3)/6.,
                        (2.-distance).clamp_min(0.)**3/6.)
    weights = one_d.prod(-1)
    pin_mass = (weights[pins]*particle_mass.to(torch.float64)[pins, None]).sum(0)
    occupied = grid_mass > 1e-12
    velocity = grid_momentum.to(torch.float64)/torch.where(occupied, grid_mass, 1.).to(torch.float64)[:, None]
    velocity = velocity+prm.dt*x.new_tensor(prm.f_ext)
    # Same separating domain-wall rule, expressed as vector masks.
    wall_branches = []
    for axis, size in enumerate(shape):
        blocked = ((indices[:, axis] < 2) & (velocity[:, axis] < 0)
                   | (indices[:, axis] >= size-2) & (velocity[:, axis] > 0))
        wall_branches.append(blocked)
        velocity[:, axis] = torch.where(blocked, 0., velocity[:, axis])
    floor_active = (nodes[:, 1] < prm.floor_y) & (velocity[:, 1] < 0) & occupied
    if bool(floor_active.any()):
        raise ValueError('Independent pin-collider fixture requires inactive floor contact')
    field = pin_mass.reshape(shape)
    components = []
    for axis in range(3):
        center = [slice(None)]*3; high = center.copy(); low = center.copy()
        center[axis], high[axis], low[axis] = slice(1, -1), slice(2, None), slice(None, -2)
        component = torch.zeros_like(field)
        component[tuple(center)] = field[tuple(high)]-field[tuple(low)]
        components.append(component.reshape(-1))
    gradient = torch.stack(components, dim=1)
    length = gradient.norm(dim=1)
    normal = gradient/torch.where(length > 1e-12, length, 1.)[:, None]
    normal_velocity = (velocity*normal).sum(1)
    contact = occupied & (pin_mass > 1e-12) & (length > 1e-12) & (normal_velocity > 0)
    grid_velocity = velocity-torch.where(contact, normal_velocity, 0.)[:, None]*normal
    grid_velocity = torch.where(occupied[:, None], grid_velocity, 0.)
    particle_velocity = weights@grid_velocity
    speed = particle_velocity.norm(dim=1)
    if prm.v_max > 0 and bool(((speed > prm.v_max) & ~pins).any()):
        raise ValueError('Independent pin-collider fixture requires inactive v_max clamp')
    # eta_mode/eta_sym only damp the APIC C output in k_g2p; they do not change v.
    particle_velocity = torch.where(pins[:, None], 0., particle_velocity)
    return dict(velocity=particle_velocity, grid_velocity=grid_velocity, pin_mass=pin_mass,
                branches=dict(occupied=occupied, contact=contact, normal_nonzero=length > 1e-12,
                              pin_support=weights[pins] > 0, pin_mass_nonzero=pin_mass > 1e-12,
                              pin_cells=torch.floor((x[pins]-x.new_tensor(prm.grid_min))/prm.dx),
                              walls=torch.stack(wall_branches), floor=floor_active))
