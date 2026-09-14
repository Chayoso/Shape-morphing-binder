"""Native raster FD for one/three splats away from alpha/tile/visibility transitions."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import socket
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from physmorph.pipeline.gauss_loss import GaussViews, _gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, choices=(1, 3), default=1)
    args = ap.parse_args()
    if socket.gethostname() != "hyde06":
        raise SystemExit("Native raster diagnostics run on hyde06 only")
    torch.set_num_threads(4)
    bundle = GaussViews([(0., 0.)], 1.8, .16, 512, "cuda")
    x = torch.tensor([[.05, -.03, .12], [-.03, .02, -.06], [.01, .04, -.2]][:args.count],
                       device="cuda", requires_grad=True)
    fg = torch.tensor([[[1.12, .04, .02], [.01, .9, .03], [0., .02, 1.04]]],
                      device="cuda").repeat(args.count, 1, 1).requires_grad_()
    def image_at(xx, ff):
        return bundle._render(xx, bundle.cams[0], ff)
    image = image_at(x, fg)
    q = torch.zeros_like(image)
    yy, xx = torch.meshgrid(torch.arange(32, device="cuda"),
                            torch.arange(32, device="cuda"), indexing="ij")
    q[:, 240:272, 240:272] = 1 + xx/32. + yy/64.
    q /= q.sum()
    # This ROI is well inside the splat's support; all other pixels have q=0.
    # Uniform color=.35, white background => image=1-.65*alpha for one splat.
    with torch.no_grad():
        alpha_roi = torch.stack([(1-image_at(x[i:i+1], fg[i:i+1])[:, 240:272, 240:272])/.65
                                 for i in range(args.count)])
    assert float(alpha_roi.min()) > .05, "ROI too close to native alpha threshold"
    minimum_transmittance = float((1-alpha_roi).prod(0).min())
    assert minimum_transmittance > .0002, "ROI too close to native early stopping"
    value = (image*q).sum()
    gradients = torch.autograd.grad(value, (x, fg))
    rows = []
    for index, g in enumerate(gradients):
        direction, analytic = g/g.norm(), float(g.norm())
        for eps in (.0001, .001, .003):
            plus, minus = [x.detach(), fg.detach()], [x.detach(), fg.detach()]
            plus[index] = plus[index]+eps*direction
            minus[index] = minus[index]-eps*direction
            with torch.no_grad():
                fd = float(((image_at(*plus)-image_at(*minus)).double()*q.double()).sum()/(2*eps))
            rows.append({"endpoint": "x" if index == 0 else "F_geom", "epsilon": eps,
                         "analytic": analytic, "fd": fd,
                         "relative_error": abs(fd-analytic)/max(abs(fd), abs(analytic), 1e-12)})
    package = _gs()[0]
    extension = importlib.import_module(package.__name__+"._C")
    result = {"N": args.count, "resolution": bundle.res, "sigma0": .16, "opacity": .9,
              "ROI": [240, 272, 240, 272], "min_ROI_alpha": float(alpha_roi.min()),
              "min_ROI_transmittance": minimum_transmittance,
              "checks": rows, "torch": torch.__version__,
              "raster_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "interior ROI with stable ordering; does not certify visibility/alpha branch transitions"}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
