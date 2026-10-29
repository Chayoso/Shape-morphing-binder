"""sc_assets.py MESH [...] -- SHOWCASE: a survey of the asset meshes for choosing the 300k showcase (CPU only, reads
nothing of any run). Per mesh, oriented as the pipeline orients it (physmorph.sampling.orientation): vertices, faces,
watertight, Euler number, connected bodies, the genus of a watertight mesh ((2 - chi) / 2 per body), the y-up extents
at the pipeline's normalisation (bounding-box diagonal 8 wu) and the filled volume there (the 110^3 fill the volume
match reads), the volume against the default sphere's (the target of a run from this source is scaled to it), and
the ppc the shape's cell would give at 300k (dx = 8 / 26, ppc = N dx^3 / V)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: F401,E402
import numpy as np  # noqa: E402
from physmorph.sampling.mesh import load_mesh, filled_volume  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402


def survey(path):
    m = load_mesh(path)
    o = orient_name(path)
    if o != "id":
        m.vertices = np.asarray(m.vertices, np.float64) @ rotation(o).T
    ext = np.asarray(m.extents, np.float64)
    s = 8.0 / float(np.linalg.norm(ext))
    vol = float(filled_volume(m)) * s ** 3
    bodies = m.split(only_watertight=False)
    wt = bool(m.is_watertight)
    genus = None
    if wt:
        genus = [int(round((2 - b.euler_number) / 2)) for b in bodies]
    return dict(mesh=Path(path).stem, orient=o, verts=int(len(m.vertices)), faces=int(len(m.faces)), watertight=wt,
                euler=int(m.euler_number), bodies=len(bodies), genus=genus,
                extents_diag8=[round(float(e * s), 3) for e in ext], volume_diag8=round(vol, 3),
                ppc300k=round(300000 * (8.0 / 26) ** 3 / vol, 1))


ref = None
for p in sys.argv[1:]:
    try:
        r = survey(p)
    except Exception as e:  # a mesh that does not load is listed, not fatal
        r = dict(mesh=Path(p).stem, error=repr(e))
    if r.get("mesh") == "isosphere":
        ref = r["volume_diag8"]
    print(json.dumps(r), flush=True)
