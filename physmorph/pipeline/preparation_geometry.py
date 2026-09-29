"""Shared window-start geometry, independent of control optimization.

These helpers preserve the ordinary runner's fragment/rest-length policy and
optimizer's reference-spacing layer construction. They do not prepare arrival,
render gates, materials, assimilation or pin admission.
"""
from physmorph.compute import array_api as np, sample_indices
from ..mpm.state import MPMParams
from .config import disc_ref_factor


_CONNECTIVITY_26 = (((1, 1, 1),) * 3,) * 3  # Fixed topology, not device state.


def fragment_mask(x: np.ndarray, prm: MPMParams) -> np.ndarray:
    """True where the particle's occupied grid cell belongs to a connected component
    (26-connectivity) of occupied cells that is NOT the largest one: material that has
    broken off the body (numerical fracture debris). Thin features stay connected to
    the body through occupied cells and are never flagged."""
    from physmorph.compute import ndimage
    ijk = np.floor((x - np.asarray(prm.grid_min, np.float32)) / prm.dx).astype(np.int64)
    dims = np.array([prm.nx, prm.ny, prm.nz])
    ok = ((ijk >= 0) & (ijk < dims)).all(1)
    occ = np.zeros((prm.nx, prm.ny, prm.nz), bool)
    occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] = True
    # STENCIL connectivity: two particles couple through shared grid nodes when their cells
    # are within the 4^3 B-spline support of each other, so components are taken on the
    # occupancy dilated by one cell (a thin feature with a one-cell occupancy gap is still one
    # body; v4 on the raw occupancy flagged a 732-particle dragon spine as a fragment)
    occ_d = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3), bool))
    # CuPy label parses topology on the host; a device-created constant would
    # cause an unnecessary array download. Occupancy and labels stay on device.
    lab, n = ndimage.label(occ_d, structure=_CONNECTIVITY_26)
    if n <= 1:
        return np.zeros(len(x), bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    body = int(sizes.argmax())
    frag = np.ones(len(x), bool)
    frag[ok] = lab[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] != body
    return frag


def prepare_bonds(x, neighbors, previous_rest, prm):
    """Refresh coupled rows; disconnected rows retain their prior material rest."""
    distance = np.linalg.norm(x[neighbors] - x[:, None, :], axis=2).astype(np.float32)
    fragmented = fragment_mask(x, prm)
    rest = (distance if previous_rest is None else
            np.where(fragmented[:, None], previous_rest, distance).astype(np.float32))
    return rest, fragmented


def prepare_layer_geometry(x, cfg):
    """Return the base five-field layer and reference spacing; no u gate."""
    if not (cfg.layer_relax or cfg.layer_ctrl):
        return None, None
    from physmorph.compute import KDTree
    from ..render.surface_recon import layer_relax_data
    n = len(x)
    subset = x[sample_indices(n, min(n, 20000))]
    spacing = float(np.median(KDTree(subset).query(subset, k=9, workers=-1)[0][:, -1])) * (min(n, 20000) / n) ** (1.0 / 3.0)
    factor = disc_ref_factor(n, cfg)
    spacing *= factor
    mask, normal, neighbors, weights = layer_relax_data(x, spacing,
        k=int(round(cfg.layer_k * factor ** 3)), h_sp=cfg.layer_h_sp,
        k_asym=int(round(32 * factor ** 3)))
    fraction = (cfg.layer_frac if cfg.layer_frac > 0 else 1.0 / float(cfg.T)) if cfg.layer_relax else 0.0
    return (mask, normal, neighbors, weights, float(fraction)), spacing
