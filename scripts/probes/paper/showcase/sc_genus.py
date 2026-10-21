"""sc_genus.py N MESH [...] -- SHOWCASE: the topology of each mesh's volume as the pipeline's sampler fills it (CPU
only): the stratified fill that holds N particles (physmorph.sampling.mesh._stratified_fill, oriented as load_normalized
orients it), as a solid of closed voxels; its Euler characteristic chi = V - E + F - C (cubical complex), its pieces b0
(26-connected voxels) and cavities b2 (6-connected empty components but the outside), and the tunnels b1 = b0 + b2 - chi
(the genus of a single piece without cavities). The topology the body has to reach at that sampling resolution."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: F401,E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from physmorph.sampling.mesh import load_mesh, _stratified_fill  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402


def euler(M):
    M = np.pad(M, 1)
    C = int(M.sum())
    # faces: unit squares touching a filled voxel, per axis (a face is shared by the 2 voxels along its normal)
    F = sum(int((M | np.roll(M, 1, ax)).sum()) for ax in range(3))
    # edges: per axis, the edge is shared by the 4 voxels around it
    E = 0
    for ax in range(3):
        a, b = [d for d in range(3) if d != ax]
        m = M | np.roll(M, 1, a)
        m = m | np.roll(m, 1, b)
        E += int(m.sum())
    # vertices: shared by the 8 voxels around it
    m = M.copy()
    for ax in range(3):
        m = m | np.roll(m, 1, ax)
    V = int(m.sum())
    return V - E + F - C


n = int(sys.argv[1])
for p in sys.argv[2:]:
    mesh = load_mesh(p)
    o = orient_name(p)
    if o != "id":
        mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
    hi, centers, pitch = _stratified_fill(mesh, n)
    ijk = np.round((centers - centers.min(0)) / pitch).astype(int)
    M = np.zeros(ijk.max(0) + 1, bool)
    M[tuple(ijk.T)] = True
    chi = euler(M)
    b0 = ndimage.label(M, structure=np.ones((3, 3, 3)))[1]
    b2 = ndimage.label(~np.pad(M, 1))[1] - 1
    print(json.dumps(dict(mesh=Path(p).stem, n=n, fill=hi, voxels=int(M.sum()), chi=chi, pieces=b0, cavities=b2,
                          tunnels=b0 + b2 - chi)), flush=True)
