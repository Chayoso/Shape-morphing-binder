"""Live VBD-MPM viewer server — the sim runs HERE (hyde06), the browser watches.

Zero dependencies beyond the repo (stdlib HTTP; the page polls a binary /state).
Per commit the runner hook publishes positions + per-splat covariance (sigma0^2 F F^T)
+ telemetry; the fixed-view client (physmorph/viewer/live.html) draws projected
covariance ellipses — the actual 3DGS quantity, no orbit, no external assets.

Run (hyde06, free GPU):
  CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/live_viewer.py \
      --n 5000 --sweeps 15 --port 8765 > output/live_viewer.log 2>&1 < /dev/null &
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
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.runner_vbd import run_vbd_pipeline  # noqa: E402
from physmorph.render.covariance import cov_from_F, sigma0_from_nn  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402

_TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])      # upper-triangle index of a 3x3


def pack_state(seq: int, rec: dict, x: np.ndarray, cov: np.ndarray,
               nodes: np.ndarray | None = None, nodeq: np.ndarray | None = None) -> bytes:
    """<u32 hlen><json hdr><f32 x[N*3]><f32 cov6[N*6]>[<f32 nodes[A*3]><f32 nodeq[A*2]>].

    nodes/nodeq (optional) = active grid node positions + per-node (|u|, |gradE|) —
    the grid-quantity layer of the viewer. phase: 'commit' or 'sweep' (the optimisation
    process itself is streamed sweep-by-sweep)."""
    a = 0 if nodes is None else int(len(nodes))
    hdr = json.dumps({
        "seq": seq, "n": int(len(x)), "a": a,
        "phase": rec.get("phase", "commit"), "sweep": rec.get("sweep"),
        "commit": rec.get("animation", -1) + 1,
        "E": rec.get("loss"), "E_el": rec.get("E_el"),
        "d_vol": rec.get("d_vol"), "d_render": rec.get("d_render"),
        "lam": rec.get("lambda"), "sweeps": rec.get("sweeps"),
        "gnorm": rec.get("gnorm"), "gnorm0": rec.get("gnorm0"),
        "move": rec.get("move"), "Jmin": rec.get("Jmin"), "run": rec.get("run", 0),
    }).encode("utf-8")
    hdr += b" " * (-len(hdr) % 4)      # 4-align: JS Float32Array(buf, off) needs off % 4 == 0
    cov6 = cov[:, _TRI[0], _TRI[1]]
    out = (struct.pack("<I", len(hdr)) + hdr
           + np.ascontiguousarray(x, "<f4").tobytes()
           + np.ascontiguousarray(cov6, "<f4").tobytes())
    if a:
        out += np.ascontiguousarray(nodes, "<f4").tobytes()
        out += np.ascontiguousarray(nodeq, "<f4").tobytes()
    return out


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = b""
        self.meta = b"{}"
        self.target = b""

    def publish(self, blob):
        with self.lock:
            self.state = blob

    def snap(self):
        with self.lock:
            return self.state


def make_handler(hub: Hub, page_path: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):                    # quiet
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
                # re-read per request: view tuning iterates without server restarts
                self._send(page_path.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/meta":
                self._send(hub.meta, "application/json")
            elif self.path == "/target":
                self._send(hub.target, "application/octet-stream")
            elif self.path == "/state":
                self._send(hub.snap(), "application/octet-stream")
            else:
                self.send_error(404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--animations", type=int, default=40)
    ap.add_argument("--sweeps", type=int, default=15)
    ap.add_argument("--lambda_auto", type=float, default=0.5)
    ap.add_argument("--assim", type=float, default=0.5)
    ap.add_argument("--vbd_young", type=float, default=2e3)
    ap.add_argument("--loop", action="store_true", default=True)
    ap.add_argument("--no-loop", dest="loop", action="store_false")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src = load_normalized(args.src, args.n, args.seed)
    tgt = load_normalized(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    cfg = PipelineConfig(animations=args.animations, vbd_sweeps=args.sweeps,
                         lambda_auto=args.lambda_auto, device=args.device,
                         assim=args.assim, vbd_young=args.vbd_young,
                         hold_after_converge=False)
    sigma0 = sigma0_from_nn(src, 0.7)
    extent = float(np.abs(tgt).max()) * 1.25

    hub = Hub()
    hub.meta = json.dumps({"n": args.n, "extent": extent, "sigma0": sigma0,
                           "src": args.src, "tgt": args.tgt, "sweeps": args.sweeps,
                           "animations": args.animations}).encode()
    N = len(tgt)
    tgt_cov = cov_from_F(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)), sigma0)
    hub.target = pack_state(0, {"animation": -1}, tgt, tgt_cov)
    F0 = np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))
    hub.publish(pack_state(0, {"animation": -1, "run": 0}, src, cov_from_F(F0, sigma0)))

    page_path = (Path(__file__).resolve().parent.parent
                 / "physmorph" / "viewer" / "live.html")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(hub, page_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[live] serving http://localhost:{args.port}  (tunnel this port)", flush=True)

    seq, run_i = 0, 0
    while True:
        def cb(a, x, F, rec, _run=run_i):
            nonlocal seq
            seq += 1
            r = dict(rec); r["run"] = _run; r["phase"] = "commit"
            hub.publish(pack_state(seq, r, x, cov_from_F(F, sigma0)))

        def cb_sweep(a, si, gn, x_s, F_s, npos, nq, _run=run_i):
            nonlocal seq
            seq += 1
            r = {"animation": a, "phase": "sweep", "sweep": si, "gnorm": gn, "run": _run}
            hub.publish(pack_state(seq, r, x_s, cov_from_F(F_s, sigma0), npos, nq))
        t0 = time.time()
        res = run_vbd_pipeline(src, tgt, prm, cfg, log=lambda *a: print(*a, flush=True),
                               on_commit=cb, on_sweep=cb_sweep)
        print(f"[live] run {run_i} finished in {time.time()-t0:.1f}s "
              f"(converged={res['converged']}); {'restarting' if args.loop else 'holding'}",
              flush=True)
        if not args.loop:
            while True:
                time.sleep(3600)
        run_i += 1
        time.sleep(2.0)                               # brief hold on the final shape


if __name__ == "__main__":
    main()
