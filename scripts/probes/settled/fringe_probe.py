"""fringe_probe.py OUT_NPZ ARCHIVE_NPZ:RAW:RUN_LOG [...] — D42: is the feathered fringe on thin features the detached sets?

For each given frame: the particles that are not connected to the body within one layer spacing (layer.detached_groups;
the body is the largest set), their number by the display's support (rendered or not), by distance to the target
(within the berth of 1.97 target spacings, to one loss cell, beyond), and by the local feature thickness of their
nearest target point (below 2 MPM cells, 2 to 4, 4 and more) against each class's share of the target's surface points.

OUT_NPZ gets two frames per input, for the display only: the frame as it is, and the frame with every detached particle
put on a random particle deep inside the body (its 32 neighbours all around it, nearest target point 4 cells thick or
more), where nothing of it is seen. Rendered side by side they show what the detached sets draw."""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.support import live_support, surface_particles  # noqa: E402
from physmorph.thin import local_thickness                     # noqa: E402

dev = torch.device("cuda")
out, frames, tgt_np = [], [], None
gen = torch.Generator(device=dev).manual_seed(0)
for spec in sys.argv[2:]:
    path, raw, log = spec.split(":")
    z = np.load(path, allow_pickle=True)
    raws = [int(v) for v in z["raws"]] if "raws" in z.files else None
    x = torch.as_tensor(np.asarray(z["frames"][raws.index(int(raw)) if raws else int(raw)], np.float32), device=dev)
    tgt_np = np.asarray(z["tgt"], np.float32)
    tgt = torch.as_tensor(tgt_np, device=dev)
    dx = float(re.search(r"\| dx=([0-9.]+) dt=", open(log).read()).group(1))
    with torch.inference_mode():
        td, tnb = knn_self_torch(tgt, 33)
        sp, cov = float(td[:, 1].median()), float(td[:, 8].median())
        h = (local_thickness(tgt, sp) / dx).float()
        tsurf = surface_particles(tgt, tnb, cov)
        d, nb = knn_self_torch(x, 33)
        group, size = detached_groups(d, nb, layer_spacing(x))
        det = group > 0
        shown = live_support(d, cov, sp) > 0
        dist, near = gpu.KNN(tgt).query(x, 1)
        dist, hx = dist[:, 0].float() / sp, h[near[:, 0]]
        deep = torch.nonzero(~det & ~surface_particles(x, nb, cov) & (hx >= 4)).squeeze(1)
        moved = x.clone()
        moved[det] = x[deep[torch.randint(len(deep), (int(det.sum()),), device=dev, generator=gen)]]
    r = det & shown
    cls = (("below 2 cells", hx < 2, h < 2), ("2 to 4 cells", (hx >= 2) & (hx < 4), (h >= 2) & (h < 4)), ("4 cells and more", hx >= 4, h >= 4))
    print(f"{path.split('/')[-2]} raw {raw}: {int(det.sum())} detached particles in {int(group.max())} sets ({int(r.sum())} rendered); rendered ones within the berth "
          f"{int((r & (dist <= 1.97)).sum())}, to one loss cell {int((r & (dist > 1.97) & (dist <= 4.4)).sum())}, beyond {int((r & (dist > 4.4)).sum())}; "
          f"distance to the target, spacings: median {float(dist[r].median()):.2f}, 90 % {float(dist[r].quantile(.9)):.2f}")
    print("   by the thickness of the nearest target point: " + "; ".join(
        f"{name}: {100 * float((r & m).sum()) / max(int(r.sum()), 1):.0f} % of them, {1e3 * float((r & m).sum()) / max(int(m.sum()), 1):.1f} per 1000 particles of the class "
        f"(the class holds {100 * float((tsurf & mt).sum()) / float(tsurf.sum()):.0f} % of the target's surface points)" for name, m, mt in cls))
    frames += [x.cpu().numpy(), moved.cpu().numpy()]
np.savez(sys.argv[1], frames=np.stack(frames), tgt=tgt_np, deliver_n=len(frames))
print("wrote", sys.argv[1], len(frames), "frames: each input as it is, then without its detached sets")
