"""The render gradient's videos (scripts/render_gradient_viz.py, RGV) on a small synthetic run: every panel, view and
resolution asked writes one frame per kept frame in its own folder (or OUT/frames when only one is asked, the hyde01
job's layout), at the asked size, and render_deliverable.py hands +viz=... to it (off by default there)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _viz_npz(path, n=6000, seed=3):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    x0 = (d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0, 1, (n, 1)) ** (1 / 3)).astype(np.float32)
    frames = np.stack([x0 * s for s in (1.0, 1.03, 1.06)]).astype(np.float32)
    W = 2
    layer = np.broadcast_to(np.linalg.norm(x0, axis=1) > 0.85, (W, n)).copy()
    pos = lambda: rng.uniform(1e-3, 1.0, (W, n)).astype(np.float32)   # noqa: E731
    scal = {k: np.full(W, v) for k, v in (("w_anim", 0.0), ("w_lam", 0.1), ("w_gshare", 1 / 3), ("w_ratio_layer", 0.3),
                                           ("w_ddfc_rel_layer", 0.2),
                                           ("w_du_rms_layer", 0.1), ("w_u0_rms_layer", 0.2),
                                           ("w_gphys_p99_layer", 1.0), ("w_dfc0_p99_layer", 1.0),
                                           ("w_layer_frac", 0.1), ("w_uactive_frac", 0.5), ("w_graw_cos", -0.1),
                                           ("w_tot_rend", 1.0), ("w_tot_phys", 3.0))}
    scal.update(w_t0=np.array([0.0, 1 / 6]), w_t1=np.array([1 / 6, 1 / 3]))     # two windows of 40 steps
    scales = dict(v1_lo=1e-3, v1_hi=1.0, v2_lo=1e-3, v2_hi=1.0, v3_max=0.5, glyph_ref=0.1, v1w_lo=1e-3, v1w_hi=1.0,
                  v2w_lo=1e-3, v2w_hi=1.0, b_lo=1e-4, b_hi=1.0, pitch=0.05)
    np.savez(path, frames=frames, raws=np.array([0, 36, 72]), tgt=x0, deliver_n=np.int64(73),
             pitch=np.float64(0.05), win_of_frame=np.array([0, 0, 1]), g_rend=pos(), g_phys=pos(), gu_rend=pos(),
             layer=layer, ddfc=rng.normal(0, .05, (W, n, 9)).astype(np.float32),
             du=rng.normal(0, .2, (W, n)).astype(np.float32), dx=pos(),
             ugate=(rng.uniform(0, 1, (W, n)) > .5).astype(np.float32), dfc0n=pos(),
             cosrp=rng.uniform(-1, 1, (W, n)).astype(np.float32),
             scales=np.str_(json.dumps(scales)), **scal)
    np.savez(str(path).replace(".npz", "_dense.npz"), tgt=x0)
    return str(path), str(path).replace(".npz", "_dense.npz")


@pytest.fixture
def run(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    pytest.importorskip("diff_gauss")
    return _viz_npz(tmp_path / "toy300k_viz.npz")


def test_every_kind_view_and_resolution_writes_its_frames(run, tmp_path):
    """Interpolated in time (the default) and held (+interp=0)."""
    sys.path.insert(0, str(ROOT))
    from scripts.render_gradient_viz import main
    viz, dense = run
    out = tmp_path / "out"
    main(["render_gradient_viz.py", viz, dense, str(out), "+viz=A,B,C,D,U", "+views=35,90", "+res=180"])
    for kd in ("A", "B", "C", "D", "U", "grid", "gridhalf"):
        for az in (35, 90):
            d = out / f"{kd}_{az}_180"
            assert sorted(os.listdir(d)) == ["0000.jpg", "0001.jpg", "0002.jpg"], d
    from PIL import Image
    assert Image.open(out / "A_35_180" / "0001.jpg").size == (320, 180)
    assert Image.open(out / "grid_35_180" / "0001.jpg").size == (640, 360)
    held = tmp_path / "held"
    main(["render_gradient_viz.py", viz, dense, str(held), "+viz=U", "+az=35", "+res=180", "+interp=0"])
    assert sorted(os.listdir(held / "frames")) == ["0000.jpg", "0001.jpg", "0002.jpg"]


def test_render_deliverable_hands_viz_on(run, tmp_path):
    viz, dense = run
    out = tmp_path / "one"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_deliverable.py"), viz, dense, str(out),
                        "+viz=D", "+az=90", "+res=180", "+fit=0"], capture_output=True, text=True, timeout=900,
                       cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    assert sorted(os.listdir(out / "frames")) == ["0000.jpg", "0001.jpg", "0002.jpg"]
