"""Live viewer server for the MAIN LINE (dynamic elasto + render-adjoint pipeline).

The sim runs HERE (hyde06); the browser watches through an SSH tunnel. Streams:
  * every ACCEPTED optimiser iteration (phase 'iter'): x_T, F_T + window telemetry —
    the optimisation process itself;
  * every promoted COMMIT (phase 'commit'): full telemetry + the loss-grid physical
    distributions (per-node mass and mass-weighted |v| on the D_vol grid) for the
    viewer's grid layer/histogram.

Zero dependencies beyond the repo (stdlib HTTP, binary /state).

Run (hyde06, free GPU):
  CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/live_viewer.py \
      --n 20000 --port 8765 > output/live_viewer.log 2>&1 < /dev/null &
Local machine:
  ssh -N -L 8765:localhost:8765 -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io
  -> open http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.render.covariance import cov_from_F, sigma0_from_nn  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402

_TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])      # upper-triangle index of a 3x3


def pack_state(seq: int, rec: dict, x: np.ndarray, cov: np.ndarray,
               nodes: np.ndarray | None = None, nodeq: np.ndarray | None = None) -> bytes:
    """<u32 hlen><json hdr><f32 x[N*3]><f32 cov6[N*6]>[<f32 nodes[A*3]><f32 nodeq[A*2]>]."""
    a = 0 if nodes is None else int(len(nodes))
    hdr = json.dumps({
        "seq": seq, "n": int(len(x)), "a": a,
        "phase": rec.get("phase", "commit"), "sweep": rec.get("sweep"),
        "commit": rec.get("animation", -1) + 1,
        "E": rec.get("loss"), "d_vol": rec.get("d_vol"), "d_render": rec.get("d_render"),
        "lam": rec.get("lambda"), "kin": rec.get("kin"),
        "v_absmax": rec.get("v_absmax"), "move": rec.get("move"),
        "gnorm": rec.get("grad_norm"), "dfc": rec.get("dfc_absmax"),
        "acc": rec.get("accepted"), "rej": rec.get("rejected"),
        "Jmin": rec.get("Jmin_traj", rec.get("Jmin")), "run": rec.get("run", 0),
    }).encode("utf-8")
    hdr += b" " * (-len(hdr) % 4)      # 4-align: JS Float32Array offsets must be %4==0
    cov6 = cov[:, _TRI[0], _TRI[1]]
    out = (struct.pack("<I", len(hdr)) + hdr
           + np.ascontiguousarray(x, "<f4").tobytes()
           + np.ascontiguousarray(cov6, "<f4").tobytes())
    if a:
        out += np.ascontiguousarray(nodes, "<f4").tobytes()
        out += np.ascontiguousarray(nodeq, "<f4").tobytes()
    return out


def grid_fields(x, v, gmin, ldx, res):
    """Loss-grid distributions: node positions + (mass, mass-weighted |v|) via CIC."""
    rel = (x - gmin) / ldx
    base = np.floor(rel).astype(np.int64)
    frac = rel - base
    m = np.zeros(res ** 3, np.float64)
    mv = np.zeros(res ** 3, np.float64)
    sp = np.linalg.norm(v, axis=1)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1 - frac[:, 2]
                w = wx * wy * wz
                ii = np.clip(base[:, 0] + ox, 0, res - 1)
                jj = np.clip(base[:, 1] + oy, 0, res - 1)
                kk = np.clip(base[:, 2] + oz, 0, res - 1)
                idx = (ii * res + jj) * res + kk
                np.add.at(m, idx, w)
                np.add.at(mv, idx, w * sp)
    act = np.nonzero(m > 0.5)[0]                      # occupied cells only
    i, j, k = act // (res * res), (act // res) % res, act % res
    nodes = np.stack([i, j, k], 1).astype(np.float32) * ldx + gmin
    nodeq = np.stack([(mv[act] / m[act]), m[act]], 1).astype(np.float32)   # (|v|, mass)
    return nodes, nodeq


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = b""
        self.meta = b"{}"
        self.target = b""
        self.history = []                # per-commit headers (chart backfill on page load)
        self.restart = threading.Event()

    def publish(self, blob, hdr=None):
        with self.lock:
            self.state = blob
            if hdr is not None:
                self.history.append(hdr)
                if len(self.history) > 500:
                    self.history.pop(0)

    def snap(self):
        with self.lock:
            return self.state

    def hist_json(self):
        with self.lock:
            return json.dumps(self.history).encode()


def make_handler(hub: Hub, page_path: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(page_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/meta":
                self._send(hub.meta, "application/json")
            elif self.path == "/target":
                self._send(hub.target, "application/octet-stream")
            elif self.path == "/state":
                self._send(hub.snap(), "application/octet-stream")
            elif self.path == "/history":
                self._send(hub.hist_json(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/restart":
                hub.restart.set()
                self._send(b'{"ok":true}', "application/json")
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--animations", type=int, default=30)
    ap.add_argument("--lambda_auto", type=float, default=0.5)
    ap.add_argument("--w_kin", type=float, default=5.0)
    ap.add_argument("--loop", action="store_true")     # default: ONE run, then hold
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src = load_normalized(args.src, args.n, args.seed)
    tgt = load_normalized(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    cfg = PipelineConfig(T=args.T, iters=args.iters, animations=args.animations,
                         lambda_auto=args.lambda_auto, w_kin=args.w_kin,
                         device=args.device, hold_after_converge=False)
    sigma0 = sigma0_from_nn(src, 0.7)
    extent = float(np.abs(tgt).max()) * 1.25
    gmin = np.asarray(prm.grid_min, np.float32)
    dmax = gmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - gmin).max() / cfg.loss_res)

    hub = Hub()
    hub.meta = json.dumps({"n": args.n, "extent": extent, "sigma0": sigma0,
                           "src": args.src, "tgt": args.tgt, "T": args.T,
                           "iters": args.iters, "animations": args.animations,
                           "pipeline": "dynamic (elasto + render adjoint)"}).encode()
    N = len(tgt)
    eye = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    hub.target = pack_state(0, {"animation": -1}, tgt, cov_from_F(eye, sigma0))
    hub.publish(pack_state(0, {"animation": -1, "run": 0}, src, cov_from_F(eye, sigma0)))

    page_path = (Path(__file__).resolve().parent.parent
                 / "physmorph" / "viewer" / "live.html")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub, page_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[live] serving http://localhost:{args.port}  (tunnel this port)", flush=True)

    seq, run_i = 0, 0
    while True:
        commit_no = {"a": -1}

        def cb_iter(it, xT, FT, tele, _run=run_i):
            nonlocal seq
            seq += 1
            r = {"animation": commit_no["a"] + 1, "phase": "iter", "sweep": it,
                 "run": _run, **{k: tele.get(k) for k in
                                 ("loss", "d_vol", "d_render", "lambda", "kin")},
                 "grad_norm": tele.get("grad_norm")}
            hub.publish(pack_state(seq, r, xT, cov_from_F(FT, sigma0)))

        def cb_commit(a, x, F, v, rec, _run=run_i):
            nonlocal seq
            seq += 1
            commit_no["a"] = a
            nodes, nodeq = grid_fields(x, v, gmin, ldx, cfg.loss_res)
            r = dict(rec); r["run"] = _run; r["phase"] = "commit"
            hdr = {"run": _run, "commit": a + 1, "E": rec.get("loss"),
                   "d_vol": rec.get("d_vol"), "d_render": rec.get("d_render"),
                   "lam": rec.get("lambda"), "kin": rec.get("kin"),
                   "move": rec.get("move"), "Jmin": rec.get("Jmin_traj", rec.get("Jmin"))}
            hub.publish(pack_state(seq, r, x, cov_from_F(F, sigma0), nodes, nodeq), hdr)

        t0 = time.time()
        res = run_pipeline(src, tgt, prm, cfg, log=lambda *a: print(*a, flush=True),
                           on_commit=cb_commit, on_iter=cb_iter)
        print(f"[live] run {run_i} finished in {time.time()-t0:.1f}s "
              f"(converged={res['converged']}); "
              f"{'looping' if args.loop else 'holding — POST /restart or the viewer button'}",
              flush=True)
        if not args.loop:
            hub.restart.wait()                        # viewer restart button
            hub.restart.clear()
        with hub.lock:
            hub.history = []
        run_i += 1
        time.sleep(1.0)


if __name__ == "__main__":
    main()
