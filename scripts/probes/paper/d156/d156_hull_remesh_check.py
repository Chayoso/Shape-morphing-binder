import sys, json, numpy as np
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d156"); sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import d156_photohull as PH
from d156_photohull import D7, o3d, trimesh, marching_cubes
from physmorph.sampling.mesh import _oriented, load_normalized, sample_in_frame, stratified_draws, fill_volume_in_frame
A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/assets_hull"
for mesh in sys.argv[1:]:
    hp = f"{A}/{mesh}_hull18_r96.obj"
    _, v_src = load_normalized("assets/isosphere.obj", 40000, 97, return_volume=True, sample="stratified")
    fr = {}; tgt = load_normalized(f"assets/{mesh}.obj", 40000, 98, match_volume=v_src, sample="stratified", frame=fr)
    pitch = 0.708 * D7.median_kth_spacing(tgt, 8, 20000)
    hm = _oriented(hp); Hw = (np.asarray(hm.vertices, np.float64) - fr["offset"]) * fr["scale"]
    H = trimesh.Trimesh(Hw, np.asarray(hm.faces), process=True)
    comps = H.split(only_watertight=False)
    info = dict(mesh=mesh, watertight=bool(H.is_watertight), volume=float(H.volume), components=len(comps),
                comp_volumes=sorted([float(c.volume) for c in comps])[:6], euler=int(H.euler_number))
    hh = 0.5 * pitch; lo = Hw.min(0) - 3 * pitch
    nc = np.ceil((Hw.max(0) + 3 * pitch - lo) / hh).astype(int) + 1
    G = np.stack(np.meshgrid(*[lo[i] + hh * np.arange(nc[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(H.vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(H.faces, np.uint32)))
    for ns in (1, 3, 5):
        sd = sc.compute_signed_distance(o3d.core.Tensor(G.astype(np.float32)), nthreads=6, nsamples=ns).numpy()
        occ = sc.compute_occupancy(o3d.core.Tensor(G.astype(np.float32)), nthreads=6, nsamples=ns).numpy()
        v0, f0, _, _ = marching_cubes((-sd).reshape(*nc), level=0.0, spacing=(hh, hh, hh))
        H0 = trimesh.Trimesh(v0 + lo, f0, process=True)
        info[f"remesh_ns{ns}"] = dict(volume=float(abs(H0.volume)), components=len(H0.split(only_watertight=False)),
                                     inside_share_sd=float((sd < 0).mean()), inside_share_occ=float((occ > .5).mean()),
                                     sign_disagree=float(((sd < 0) != (occ > .5)).mean()))
    # where do the sd sign and a dense occupancy disagree: share of the hull sample with sd > 0 (called outside)
    n_vh = int(round(40000 * fill_volume_in_frame(hp, fr, 40000) / fill_volume_in_frame(f"assets/{mesh}.obj", fr, 40000)))
    xs, _ = sample_in_frame(hp, fr, n_vh, 98)
    sdx = sc.compute_signed_distance(o3d.core.Tensor(xs.astype(np.float32)), nthreads=6, nsamples=3).numpy()
    info["hull_sample_called_outside_by_sd_ns3"] = float((sdx > 0.5 * pitch).mean())
    print(json.dumps(info), flush=True)
