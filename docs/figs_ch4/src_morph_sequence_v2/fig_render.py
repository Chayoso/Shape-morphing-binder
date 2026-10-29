"""fig_render.py FRAMES12_NPZ DENSE_NPZ OUT_DIR +ks=0,10,20 [+az=35] [+el=18] [+dist=1] [+min_piece=10] [+raw=1]
-- FIGS_V2 (2026-10-22, the dissertation's Figure 4.2): studio-look stills of chosen kept frames of a finished run, the
base display (no detail layer), with one display rule for small separate pieces.

The display is render_deliverable.py's of repo_t1s (tag freeze-2026-10-15-rt1) at +fit=0 +field=zbs, unchanged: the
exterior's discs of the frame's particles (Zhu-Bridson field with Solenthaler et al.'s correction, radius 3, offset 0.8
pitches) on a lattice of 0.92 of the dense sample's pitch, the base normals averaged twice over the particles' reach,
the StudioRaster camera at 3840 x 2160 (azimuth, elevation, distance x 3.6 radii).

The display rule (display only; the particles and the run are untouched): the drawn discs are split into connected
pieces (discs within two lattice steps of each other are joined); each particle is given to the piece of its nearest
disc; a piece that holds fewer than MIN_PIECE particles is not drawn. Everything else is drawn as before, so a frame
without such pieces is the same picture as render_deliverable.py's. A second piece of MIN_PIECE particles or more stays
drawn and is reported (size, gap to the largest piece in pitches).

Per frame: OUT_DIR/frames/NNNN.jpg (rule applied) and, with +raw=1, OUT_DIR/frames_raw/NNNN.jpg (all discs); one JSON
line in OUT_DIR/pieces.jsonl: the pieces, the particles and discs left out, the secondary pieces kept.
The images are the raster's own (mirrored); flip them left-right when encoding (as SHOWCASE does)."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_t1s")
from physmorph import gpu                                      # noqa: E402  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image                                          # noqa: E402
from scipy.sparse import coo_matrix                            # noqa: E402
from scipy.sparse.csgraph import connected_components          # noqa: E402
from scipy.spatial import cKDTree                              # noqa: E402
from physmorph.render.exterior import Lattice, make_field      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402

dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
RAD, OFF, FIELD = 3., .8, "zbs"
AZ, EL, DIST = float(opts.get("az", 35.)), float(opts.get("el", 18.)), float(opts.get("dist", 1.))
MIN_PIECE = int(opts.get("min_piece", 10))
RAW = opts.get("raw", "0") == "1"
DRAW = opts.get("draw", "1") == "1"                         # +draw=0: the piece record only
PJ = opts.get("pieces_out", "pieces.jsonl")
z = np.load(argv[1], allow_pickle=True)
frames = z["frames"]
n_frames = len(frames)
KS = sorted({n_frames - 1 if v == "last" else int(v) for v in opts["ks"].split(",")})
dense = torch.as_tensor(np.asarray(np.load(argv[2])["tgt"], np.float32), device=dev)
out = Path(argv[3])
(out / "frames").mkdir(parents=True, exist_ok=True)
if RAW:
    (out / "frames_raw").mkdir(parents=True, exist_ok=True)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731
a, a_c = pitch(tgt), pitch(dense)
del dense
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
h_lat = 0.92 * a_c
lat = Lattice(center, 2.8 * radius)
cam = StudioRaster(center, radius * DIST, 3840, 2160, AZ, EL)
print(f"pitch {a:.4f}, dense {a_c:.4f}, lattice {h_lat:.4f}; cam az {AZ} el {EL} dist {DIST}; frames {KS} of {n_frames}; "
      f"min_piece {MIN_PIECE}", flush=True)


def base(x):
    """render_deliverable.py's Surface at h = 0: the discs and their twice-averaged base normals."""
    with torch.no_grad():
        p, g, _, _ = lat.discs(make_field(FIELD, x, a, radius=RAD, offset=OFF, device_field=True), h_lat)
        n = nnf.normalize(g, dim=1)
        d, nbx = knn_self_torch(x, 33)
        reach = float(d[(x - x[nbx[:, 1:]].mean(1)).norm(dim=1) >= .5 * a / .708, 32].median())
        spacing = float(knn_self_torch(p, 2)[0][:, 1].median())
        dd, nn = gpu.KNN(p).query(p, int(min(256, 3.63 * (reach / spacing) ** 2)) + 1)
        wr = (dd.float() <= reach)[..., None]
        for _ in range(2):
            n = nnf.normalize((n[nn] * wr).sum(1), dim=1, eps=1e-9)
    return p, n


def draw(q, n, path):
    """render_deliverable.py's draw, unchanged."""
    rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
    rot = torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)
    sig = torch.full((len(q),), h_lat, device=dev)
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    with torch.no_grad():
        img = cam(q, n, cov, torch.full((len(q),), .92, device=dev), normal_kernel=normal_filter_size(2160, False))
    Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(path, quality=93)


def pieces(p, x):
    """The display's connected pieces: labels per disc, particles per piece (by nearest disc), discs per piece."""
    kk = 16
    dd, ii = gpu.KNN(p).query(p, kk)
    dd, ii = dd.cpu().numpy(), ii.cpu().numpy()
    m = dd <= 2.0 * h_lat
    rows = np.repeat(np.arange(len(p)), kk)[m.ravel()]
    cols = ii.ravel()[m.ravel()]
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(p), len(p)))
    nc, lab = connected_components(g, directed=False)
    near = gpu.KNN(p).query(x, 1)[1].reshape(-1).cpu().numpy()
    npart = np.bincount(lab[near], minlength=nc)
    ndisc = np.bincount(lab, minlength=nc)
    return nc, lab, npart, ndisc, near


def body_set(xc):
    """D145's pieces: the largest set of particles connected at 1.5 pitches (pieces145.py), as a mask."""
    pairs = cKDTree(xc).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(xc), len(xc)))
    _, pl = connected_components(g, directed=False)
    return pl == np.argmax(np.bincount(pl))


rows_out = open(out / PJ, "w")
for k in KS:
    t0 = time.time()
    x = torch.as_tensor(np.asarray(frames[k], np.float32), device=dev)
    p, n = base(x)
    nc, lab, npart, ndisc, near = pieces(p, x)
    onb = body_set(x.cpu().numpy().astype(np.float64))
    inb = np.bincount(lab[near][onb], minlength=nc)               # particles of each piece on the body's set
    off = (npart - inb) > 0.5 * npart                              # a piece mostly off the body: floating
    big = int(np.argmax(npart))
    drop_c = np.flatnonzero(npart < MIN_PIECE)
    keep_disc = torch.as_tensor(~np.isin(lab, drop_c), device=dev)
    second = [int(c) for c in np.flatnonzero(npart >= MIN_PIECE) if c != big]
    sec = []
    if second:
        tb = gpu.KNN(p[torch.as_tensor(lab == big, device=dev)])
        for c in second:
            gap = float(tb.query(p[torch.as_tensor(lab == c, device=dev)], 1)[0].min()) / a
            sec.append(dict(particles=int(npart[c]), discs=int(ndisc[c]), gap_pitch=round(gap, 2),
                            off_body=int(npart[c] - inb[c]), floating=bool(off[c])))
    small = sorted((int(npart[c]) for c in drop_c), reverse=True)
    row = dict(frame=k, t=round(k * 0.05, 3), raw=int(z["raws"][k]), discs=len(p), pieces=int(nc),
               body_particles=int(npart[big]), dropped_pieces=len(drop_c), dropped_particles=int(npart[drop_c].sum()),
               dropped_discs=int(ndisc[drop_c].sum()), dropped_sizes=small[:40],
               dropped_floating_pieces=int(off[drop_c].sum()), dropped_floating_particles=int(npart[drop_c][off[drop_c]].sum()),
               kept_secondary=sec, kept_floating=[s for s in sec if s["floating"]])
    if DRAW:
        draw(p[keep_disc], n[keep_disc], out / "frames" / f"{k:04d}.jpg")
    if RAW and DRAW:
        draw(p, n, out / "frames_raw" / f"{k:04d}.jpg")
    row["s"] = round(time.time() - t0, 1)
    rows_out.write(json.dumps(row) + "\n")
    rows_out.flush()
    print(json.dumps(row), flush=True)
print("wrote", out, len(KS), "frames", flush=True)
