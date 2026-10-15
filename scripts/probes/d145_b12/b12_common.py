"""B12 (D145): the two render-only baselines (Large Steps, FlexiCubes), shared pieces. Runs in B12's own env
(baselines/b12/env: torch 2.8.0+cu128, nvdiffrast 0.4.0, largesteps 0.2.2, gpytoolbox 0.3.8).

  targets     B5's export of each mesh in the pipeline frame (`<name>_ourframe.ply`; the teapot's outer body
              `teapot_outer_ourframe.ply`, as B5: the sampler fills the two inner cavity shells), vertices merged,
              holes fan-filled where the mesh is open (the bunny's five base holes, as B11).
  source      the sphere at the centre of the pipeline's seed-97 source sample and the radius of the ball with the
              volume of B5's exported isosphere (B11's definition, the D145 agent's decision).
  views       arm a (matched): the pipeline's render loss as the 300k runs use it: 6 azimuths x elevations (0, 0.5,
              -0.5) rad = 18 orthographic views (physmorph.pipeline.render_loss.make_views), 188 px (96 x (N /
              40000)^(1/3), N = 300k, as pipeline_run.py), the half-extent 1.25 x max |target sample coordinate|
              (target.py). Arm b (strong): 25 orthographic views on a Fibonacci sphere (the Mitsuba 3 tutorial's
              lattice) at 512 px, the same extent. View-set seed 97 = the rig as stated; seeds 99 and 101 rotate the
              whole rig by scipy's Rotation.random(random_state=seed).
  images      the pipeline render loss's image model on the true surface: per pixel (shade x alpha, alpha), shade =
              0.25 + 0.75 max(n . d, 0) with d toward the camera (a headlight, ambient 0.25), background 0;
              nvdiffrast's rasterizer, interpolation and antialias (the silhouette's gradient), as the Large Steps
              renderer (scripts/render.py NVDRenderer) and the FlexiCubes renderer (render_mesh_paper).
"""
import json
import os

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

B12 = "/data/relcfd/chayo/physmorph_v2/baselines/b12"
B5 = "/data/relcfd/chayo/physmorph_v2/baselines/buonomo/inputs"
IND = "/data/relcfd/chayo/physmorph_v2/output/gpu/d90"
MESHES = ("bunny", "dragon", "teapot", "bob", "armadilo", "homer")
AMBIENT = 0.25
N_RUN, MASS_REF_N, RENDER_RES = 300000, 40000, 96


def target_mesh(name):
    """(trimesh in the pipeline frame, report): B5's export, merged, holes fan-filled when open."""
    fn = f"{B5}/{name}/teapot_outer_ourframe.ply" if name == "teapot" else f"{B5}/{name}/{name}_ourframe.ply"
    raw = trimesh.load(fn, process=False)
    m = trimesh.Trimesh(np.asarray(raw.vertices, np.float64), np.asarray(raw.faces, np.int64), process=True)
    rep = dict(file=fn, V=int(len(m.vertices)), F=int(len(m.faces)), watertight=bool(m.is_watertight))
    if not m.is_watertight:
        nf = len(m.faces)
        trimesh.repair.fill_holes(m, use_fan=True)
        rep.update(filled_faces=int(len(m.faces) - nf), watertight_after_fill=bool(m.is_watertight))
    rep["volume"] = float(m.volume)
    return m, rep


def source_sphere(name):
    """(centre, radius) of the source sphere in the pipeline frame (B11's definition)."""
    src = np.load(f"{B5}/{name}/pipeline_samples.npz")["src"].astype(np.float64)
    iso = trimesh.load(f"{B5}/{name}/sphere_for_{name}_ourframe.ply", process=True)
    return src.mean(0), float((3.0 * float(iso.volume) / (4.0 * np.pi)) ** (1.0 / 3.0))


def own_target_sample(name):
    return np.asarray(np.load(f"{B5}/{name}/pipeline_samples.npz")["tgt"], np.float32)


def extent_of(name):
    """The render half-extent of the pipeline (target.py): 1.25 x max |coordinate| of the run's target sample."""
    return 1.25 * float(np.abs(own_target_sample(name)).max())


def make_views(n_azim=6, elevs=(0.0, 0.5, -0.5)):
    """physmorph.pipeline.render_loss.make_views, verbatim (an azimuth ring per elevation, rings offset)."""
    views = []
    n_rings = max(len(tuple(elevs)), 1)
    for i, phi in enumerate(elevs):
        off = (i / n_rings) * (2 * np.pi / max(n_azim, 1))
        for t in np.linspace(0, 2 * np.pi, n_azim, endpoint=False):
            views.append((float(t + off), float(phi)))
    return views


def basis_from_views(views):
    """right, up, toward-camera of the pipeline's orthographic views (losses/silhouette.py _view_basis)."""
    th = np.array([v[0] for v in views])
    ph = np.array([v[1] for v in views])
    right = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], 1)
    up = np.stack([-np.sin(ph) * np.sin(th), np.cos(ph), -np.sin(ph) * np.cos(th)], 1)
    return right, up, np.cross(right, up)


def fibonacci_dirs(n):
    """The Mitsuba 3 shape-optimisation tutorial's sensor directions (golden-ratio lattice)."""
    g = (1 + 5 ** 0.5) / 2
    i = np.arange(n)
    th = 2 * np.pi * i / g
    ph = np.arccos(1 - 2 * (i + 0.5) / n)
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def basis_from_dirs(d):
    """Orthographic cameras looking at the origin from directions d with up ~ +y."""
    yw = np.array([0.0, 1.0, 0.0])
    right = np.cross(yw[None], d)
    bad = np.linalg.norm(right, axis=1) < 1e-3
    right[bad] = np.cross(np.array([0.0, 0.0, 1.0])[None], d[bad])
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    up = np.cross(d, right)
    return right, up, d.copy()


def view_set(arm, seed):
    if arm == "a":
        right, up, d = basis_from_views(make_views())
        res = int(np.ceil(RENDER_RES * max(1.0, (N_RUN / MASS_REF_N) ** (1.0 / 3.0))))
        desc = "the pipeline's 18 orthographic views (6 azimuths x elevations 0, 0.5, -0.5 rad)"
    elif arm == "b":
        right, up, d = basis_from_dirs(fibonacci_dirs(25))
        res = 512
        desc = "25 orthographic views on the Mitsuba tutorial's Fibonacci lattice"
    else:
        raise ValueError(arm)
    if seed != 97:
        R = Rotation.random(random_state=seed).as_matrix()
        right, up, d = right @ R.T, up @ R.T, d @ R.T
        desc += f", the rig rotated by Rotation.random(random_state={seed})"
    return dict(right=right, up=up, toward=d, res=res, n=len(d), desc=desc)


class Renderer:
    """(shade x alpha, alpha) images of a triangle mesh from orthographic views, nvdiffrast's CUDA rasterizer."""

    def __init__(self, vs, extent, boost=1.0, device="cuda"):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext(device=device)
        E = float(extent)
        mats = np.zeros((vs["n"], 4, 4))
        mats[:, 0, :3] = vs["right"] / E
        mats[:, 1, :3] = vs["up"] / E
        mats[:, 2, :3] = -vs["toward"] / (4.0 * E)        # nearer = smaller depth; |x . d| < 4 E for the scene
        mats[:, 3, 3] = 1.0
        self.mvp = torch.tensor(mats, dtype=torch.float32, device=device)
        self.dirs = torch.tensor(vs["toward"], dtype=torch.float32, device=device)
        self.res = int(vs["res"])
        self.boost = float(boost)

    def render(self, v, f, n, per_face=False, views=None, raw=False):
        """v (Nv,3) float32, f (Nf,3) int32, n vertex (Nv,3) or face (Nf,3) unit normals. Returns (V,res,res,2)
        antialiased (shade x alpha, alpha); with raw=True also the un-antialiased shade (V,res,res,1) and the
        coverage (V,res,res,1)."""
        dr = self.dr
        mvp = self.mvp if views is None else self.mvp[views]
        dirs = self.dirs if views is None else self.dirs[views]
        v_clip = torch.matmul(torch.nn.functional.pad(v, (0, 1), value=1.0), mvp.transpose(1, 2)).contiguous()
        rast, _ = dr.rasterize(self.ctx, v_clip, f, (self.res, self.res))
        if per_face:
            idx = torch.arange(f.shape[0], dtype=torch.int32, device=v.device)[:, None].repeat(1, 3)
            nimg, _ = dr.interpolate(n[None].contiguous(), rast, idx)
        else:
            nimg, _ = dr.interpolate(n[None].contiguous(), rast, f)
        nimg = torch.nn.functional.normalize(nimg, dim=-1)
        shade = AMBIENT + (1 - AMBIENT) * (nimg * dirs[:, None, None, :]).sum(-1, keepdim=True).clamp(min=0)
        cov = (rast[..., 3:] > 0).float()
        img = dr.antialias(torch.cat([shade * cov, cov], -1).contiguous(), rast, v_clip, f,
                           pos_gradient_boost=self.boost)
        if raw:
            return img, shade * cov, cov
        return img


def face_normals(v, f):
    a, b, c = v[f[:, 0].long()], v[f[:, 1].long()], v[f[:, 2].long()]
    return torch.nn.functional.normalize(torch.linalg.cross(b - a, c - a), dim=1)


def topo_np(V, F):
    """Euler characteristic, closedness and component count of a triangle mesh (numpy; the genus when closed and
    edge-manifold, summed over components: g = sum_k (2 - chi_k) / 2)."""
    F = np.asarray(F, np.int64)
    if len(F) == 0:
        return dict(V=0, F=0, E=0, euler=0, closed=False, boundary_edges=0, nonmanifold_edges=0, components=0,
                    genus=None)
    used = np.unique(F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    chi = int(len(used) - len(ue) + len(F))
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = int(F.max()) + 1
    g = coo_matrix((np.ones(len(ue)), (ue[:, 0], ue[:, 1])), shape=(nv, nv))
    ncomp, lab = connected_components(g, directed=False)
    ncomp = int(len(np.unique(lab[used])))
    closed = bool((cnt == 2).all())
    genus = int((2 * ncomp - chi) // 2) if closed and (2 * ncomp - chi) % 2 == 0 else None
    return dict(V=int(len(used)), F=int(len(F)), E=int(len(ue)), euler=chi, closed=closed,
                boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()), components=ncomp,
                genus=genus)


def signed_volume(V, F):
    """Enclosed signed volume (positive = outward orientation)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    if len(F) == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def pick_frames(L, cand, n=21):
    """21 frames at matched progress of the method's own image loss: q = (L_0 - L_k) / (L_0 - L_end) over the
    candidate iterates; frame j < n-1 is the first candidate with q >= j / (n-1), the last frame the end iterate."""
    L = np.asarray(L, np.float64)
    q = (L[0] - L) / max(L[0] - L[-1], 1e-30)
    picks = []
    for j in range(n - 1):
        hit = np.nonzero(q >= j / (n - 1))[0]
        picks.append(int(hit[0]) if len(hit) else len(L) - 1)
    picks.append(len(L) - 1)
    return [int(cand[i]) for i in picks], [float(q[i]) for i in picks], picks


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
