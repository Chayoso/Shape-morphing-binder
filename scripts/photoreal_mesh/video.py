"""The two drivers of the photoreal renderer: a still (--still, with --save_mesh the mesh and its
numbers for scripts/probes/surface_gt.py) and the video (per-frame reconstruction with --prefetch
threads, optional surface tracking, ffmpeg encode) with its sidecar <out>.components.txt, the
per-frame QA the videos are judged by."""
import copy
import os
import subprocess
import tempfile

import numpy as np
import open3d as o3d
import torch

from photoreal_mesh.measures import bumpiness, isolated_count
from photoreal_mesh.scene import label, render_views
from photoreal_mesh.surface import mesh_of
from photoreal_mesh.tracking import advect_vertices, closest_on


def render_still(ctx, view):
    """--still <frame> (or -2 = the target cloud through the same pipeline): one mesh, one png."""
    a, dev, tgt, frames_np, vox, spacing = ctx.a, ctx.dev, ctx.tgt, ctx.frames_np, ctx.vox, ctx.spacing
    pca_sigma_sp, bulk_voxel, bulk_particle = ctx.pca_sigma_sp, ctx.bulk_voxel, ctx.bulk_particle
    iso_frac, layer_thr, rho_bulk = ctx.iso_frac, ctx.layer_thr, ctx.rho_bulk
    fr = tgt if a.still == -2 else torch.as_tensor(np.asarray(frames_np[a.still], np.float32), device=dev)
    # --still -2 renders the TARGET cloud through the same pipeline: the floor of the bumpiness
    # measure for this discretisation
    m, n_comp, n_drop, n_bridge, n_cav = mesh_of(ctx, fr, a.still if a.still >= 0 else None)
    bump = bumpiness(m)
    print(f"[photoreal] still {a.still}: kernel {a.kernel} surface {a.surface} post {a.post}, "
          f"bumpiness (mean |dihedral|) {bump:.2f} deg, triangles {len(m.triangles) if m is not None else 0}, "
          f"components {n_comp} dropped {n_drop} cavities {n_cav} bridged {n_bridge}", flush=True)
    img = label(render_views(view, m), f"{a.label} frame {a.still}  {a.kernel}/{a.surface}/{a.post}  bump {bump:.1f} deg  "
                                 f"components {n_comp} (dropped {n_drop}, cavities {n_cav}, bridged {n_bridge})")
    o3d.io.write_image(a.out, o3d.geometry.Image(np.ascontiguousarray(img)))
    if a.save_mesh and m is not None:
        # the mesh + the discretisation it was made at, for scripts/probes/surface_gt.py
        import json
        o3d.io.write_triangle_mesh(a.save_mesh, m, write_ascii=False, compressed=True)
        with open(os.path.splitext(a.save_mesh)[0] + ".json", "w") as fh:
            json.dump({"npz": a.npz, "frame": a.still, "kernel": a.kernel, "surface": a.surface, "post": a.post,
                       "pull": a.pull, "layer": a.layer, "poisson_cell": a.poisson_cell, "poisson_trim": a.poisson_trim, "pca_sigma": pca_sigma_sp, "label": a.label, "bulk": a.bulk, "bulk_voxel_over_particle": bulk_voxel / bulk_particle,
                       "vox": vox, "spacing": spacing, "iso_frac": iso_frac, "layer_thr_frac": layer_thr / rho_bulk,
                       "bump": bump, "triangles": int(len(m.triangles)), "components": n_comp, "dropped": n_drop,
                       "cavities": n_cav, "bridged": n_bridge}, fh, indent=1)
        print(f"saved {a.save_mesh}")
    print(f"saved {a.out} (components {n_comp}, sub-cell dropped {n_drop})")


def render_video(ctx, view):
    """The video: every --stride-th archived frame (and the last), reconstructed, optionally tracked, rendered
    from --views, encoded with ffmpeg to --out; the per-frame QA numbers go to <out>.components.txt."""
    a, dev, frames_np, dn, spacing = ctx.a, ctx.dev, ctx.frames_np, ctx.dn, ctx.spacing
    bulk_voxel, bulk_particle, iso_frac, cell_wu = ctx.bulk_voxel, ctx.bulk_particle, ctx.iso_frac, ctx.cell_wu
    idx = list(range(0, dn, a.stride))
    if idx[-1] != dn - 1:
        idx.append(dn - 1)
    if a.max_frames > 0:
        idx = idx[: a.max_frames]
    tmp = tempfile.mkdtemp(prefix="photoreal_")
    qa = []
    trk = None            # (tracked mesh, particles at its last frame, drawn topology)
    prev_fresh = None     # (fresh mesh, particles) of the previous video frame — the re-fit jitter reference

    def _mesh_job(i):
        return mesh_of(ctx, torch.as_tensor(np.asarray(frames_np[i], np.float32), device=dev), i)

    _pool = None; _fut = {}
    if a.prefetch > 0 and len(idx) > 1:
        # 2026-09-23: the per-frame reconstruction (density grid on the GPU + a Poisson child process) is
        # independent across frames; run the next --prefetch frames ahead in threads so the Poisson
        # children overlap (the 128-core host was rendering one frame at a time)
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=a.prefetch)
        for j in range(min(a.prefetch, len(idx))):
            _fut[j] = _pool.submit(_mesh_job, idx[j])
    for k, i in enumerate(idx):
        x_np = np.asarray(frames_np[i], np.float32)
        if _pool is not None:
            nxt = k + a.prefetch
            if nxt < len(idx) and nxt not in _fut:
                _fut[nxt] = _pool.submit(_mesh_job, idx[nxt])
            m, n_comp, n_drop, n_bridge, n_cav = _fut.pop(k).result()
        else:
            m, n_comp, n_drop, n_bridge, n_cav = mesh_of(ctx, torch.as_tensor(x_np, device=dev), i)
        n_iso = isolated_count(x_np)
        x64 = x_np.astype(np.float64)
        jitter = drift = float("nan"); remeshed = 0; n_orphan = 0
        if m is not None and prev_fresh is not None and prev_fresh[0] is not None and len(m.triangles) > 0:
            # the re-fit jitter of an independent reconstruction per frame: the previous fresh mesh carried
            # along with the material against the current fresh mesh (spacings)
            Vp = advect_vertices(np.asarray(prev_fresh[0].vertices), prev_fresh[1], x64, a.track_k, spacing)
            jitter = float(np.linalg.norm(closest_on(m, Vp) - Vp, axis=1).mean() / spacing)
        prev_fresh = (m, x64)
        draw_m = m
        if a.track and m is not None and len(m.triangles) > 0:
            # the topology the PARTICLES confirm: drawn pieces the filament rule could not tie to the body (a piece
            # the reconstruction broke off a thin neck while the particles continue is NOT a topology change —
            # the tracked mesh keeps the neck as a tube) and the cavities
            n_drawn = n_comp - n_drop - n_cav
            topo = (max(n_drawn - n_bridge, 1),)           # cavities are interior and never drawn: not a trigger (ogre: 30 of 42 re-meshes)
            if trk is None:
                trk = (copy.deepcopy(m), x64, topo, k); remeshed = 1
            else:
                V = advect_vertices(np.asarray(trk[0].vertices), trk[1], x64, a.track_k, spacing)
                P = closest_on(m, V)
                vdist = np.linalg.norm(P - V, axis=1)
                drift = float(np.median(vdist) / spacing)                       # the surface as a whole, not a lost neck
                stretched = False
                if a.track_stretch > 0:
                    # the tracked triangles stretch where the surface grows (an ear pulled out of the body): past
                    # twice the fresh reconstruction's own edge length they cannot carry its detail (Nyquist) and
                    # read as flat facets — re-mesh there (with --track_keep the re-mesh wipes nothing)
                    def _edges(mm):
                        t = np.asarray(mm.triangles); v = np.asarray(mm.vertices)
                        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
                        return np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
                    # the reference is the fresh mesh's own long-edge tail (its 99th percentile), so the
                    # tracked mesh is allowed what a fresh reconstruction has and no more; the web of
                    # stretched triangles between two growing ears (g41 bunny frame 357) is 5-20x that
                    l0 = float(np.quantile(_edges(m), 0.99))
                    tri_t = np.asarray(trk[0].triangles)
                    e_t = np.concatenate([tri_t[:, [0, 1]], tri_t[:, [1, 2]], tri_t[:, [2, 0]]])
                    l_t = np.linalg.norm(V[e_t[:, 0]] - V[e_t[:, 1]], axis=1)
                    stretched = bool(np.quantile(l_t, 0.99) > a.track_stretch * l0)
                    tri_long = (l_t.reshape(3, -1) > a.track_stretch * l0).any(0)   # per triangle: any edge stretched
                if topo != trk[2] or drift > a.track_tol or stretched or (a.track_every > 0 and k - trk[3] >= a.track_every):
                    fresh = copy.deepcopy(m)
                    if a.track_keep and topo == trk[2]:
                        # a re-mesh that the particles did NOT ask for (drift / periodic): the fresh reconstruction
                        # may lack a neck the tracked mesh still carries as a tube (2026-09-23, the wiped
                        # connections of the g41 videos: at re-mesh frames the drawn area changed 3-9x more than
                        # elsewhere). Keep the tracked triangles that have no fresh counterpart — all three
                        # vertices farther than one spacing from the fresh surface — and let the pull merge
                        # them back when the reconstruction regains the neck. A particle-confirmed topology
                        # change still re-meshes honestly.
                        tri = np.asarray(trk[0].triangles)
                        keep = (vdist > spacing)[tri].all(1)
                        if a.track_stretch > 0:
                            keep &= ~tri_long                    # never keep a stretched (web) triangle
                        n_orphan = int(keep.sum())
                        if n_orphan > 0:
                            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(tri[keep]))
                            om.remove_unreferenced_vertices()
                            fresh += om
                            fresh.compute_vertex_normals()
                    trk = (fresh, x64, topo, k); remeshed = 1
                else:
                    gate = (vdist <= spacing)[:, None]                          # no pull where the fresh surface is absent
                    tm = trk[0]
                    tm.vertices = o3d.utility.Vector3dVector(V + a.track_alpha * (P - V) * gate)
                    tm.compute_vertex_normals()
                    trk = (tm, x64, topo, trk[3])
            draw_m = trk[0]
        n_vert = int(len(draw_m.vertices)) if draw_m is not None else 0
        qa.append((i, n_comp, n_iso, n_drop, n_bridge, n_cav, jitter, drift, remeshed, n_vert, n_orphan))
        img = label(render_views(view, draw_m), f"{a.label}  frame {i}/{dn - 1}")
        o3d.io.write_image(os.path.join(tmp, f"f{k:05d}.png"), o3d.geometry.Image(np.ascontiguousarray(img)))
        if k % 25 == 0:
            print(f"[photoreal] frame {k + 1}/{len(idx)} (archived {i}) components {n_comp} dropped {n_drop} isolated {n_iso}", flush=True)
    n = len(idx)
    for h in range(a.hold):
        os.link(os.path.join(tmp, f"f{n - 1:05d}.png"), os.path.join(tmp, f"f{n + h:05d}.png"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
                    "-movflags", "faststart", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out], check=True)
    with open(a.out + ".components.txt", "w") as fh:
        fh.write("archived_frame isosurface_components isolated_particles subcell_components_dropped components_bridged_to_body interior_cavities refit_jitter_sp track_drift_sp remeshed drawn_vertices orphan_triangles_kept\n")
        for i, n_comp, n_iso, n_drop, n_bridge, n_cav, jit, drf, rm, nv, no in qa:
            fh.write(f"{i} {n_comp} {n_iso} {n_drop} {n_bridge} {n_cav} {jit:.4f} {drf:.4f} {rm} {nv} {no}\n")
        comps = np.array([q[1] for q in qa]); isos = np.array([q[2] for q in qa]); drops = np.array([q[3] for q in qa])
        bridges = np.array([q[4] for q in qa]); cavs = np.array([q[5] for q in qa])
        jits = np.array([q[6] for q in qa], float); drfs = np.array([q[7] for q in qa], float); rms = np.array([q[8] for q in qa])
        drawn = comps - drops - cavs
        fh.write(f"# re-fit jitter of the independent per-frame reconstruction (previous fresh mesh carried with the material vs "
                 f"the current fresh mesh, spacings): mean {np.nanmean(jits):.3f}, p90 {np.nanpercentile(jits, 90):.3f}; "
                 f"tracking {'ON' if a.track else 'off'}"
                 + (f": re-meshed {int(rms.sum())} frames (particle-confirmed topology change, drift > {a.track_tol} sp or every {a.track_every} frames), tracked drift (median) before the pull "
                    f"mean {np.nanmean(drfs):.3f} sp, pull alpha {a.track_alpha}, k {a.track_k}" if a.track else "") + "\n")
        fh.write(f"# interior cavities (closed surfaces with the sign opposite to the body, removed, not pieces): "
                 f"{(cavs > 0).sum()} frames (max {cavs.max()})\n")
        fh.write(f"# filament bridges (particle connectivity): {(bridges > 0).sum()} frames with a drawn component tied to the "
                 f"body by particles the isosurface does not enclose (max {bridges.max()}); drawn components>1 AND not bridged "
                 f"in {((drawn > 1) & (bridges < drawn - 1)).sum()} frames\n")
        fh.write(f"# surface {a.surface}" + (f" (outer layer by {a.layer}, pull {a.pull}, octree cell {a.poisson_cell} spacing, "
                 f"trim {a.poisson_trim} sigma; docs/method.md 10.12); Poisson fallback to the level set in "
                 f"{len(ctx.fallback_frames)} frames {ctx.fallback_frames[:20]}" if a.surface == "poisson" else "") +
                 f"; bulk = {a.bulk} median (voxel/particle {bulk_voxel / bulk_particle:.3f})\n")
        fh.write(f"# iso {iso_frac:.3f} x bulk ({'auto: two-particle filament level' if str(a.iso).lower() == 'auto' else 'fixed'}), "
                 f"blur {a.blur} spacings, grid {a.grid}\n")
        fh.write(f"# frames {len(qa)}  raw components>1 in {(comps > 1).sum()} frames (max {comps.max()})  "
                 f"drawn components>1 in {(drawn > 1).sum()} frames (max {drawn.max()})  "
                 f"sub-cell components dropped in {(drops > 0).sum()} frames (cell {cell_wu:.3f} wu, min {a.min_cells:g} cells)  "
                 f"isolated particles max {isos.max()} (frame {idx[int(isos.argmax())]})\n")
    for fpath in os.listdir(tmp):
        os.remove(os.path.join(tmp, fpath))
    os.rmdir(tmp)
    print(f"saved {a.out} ({n} frames + {a.hold} hold; raw components>1 in {(comps > 1).sum()}/{len(qa)} frames, "
          f"drawn components>1 in {(drawn > 1).sum()}, max isolated particles {isos.max()})")
