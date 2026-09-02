"""Live-viewer server, embeddable in ANY run (live_viewer.py CLI or pipeline_run.py
--live_port): streams accepted iterations + commits over stdlib HTTP for live.html,
and serves /quad — the 2x2 dashboard that embeds four ports (one per GPU) so a full
4-GPU batch is watchable in real time from one page.

Binary /state protocol: <u32 hlen><json hdr><f32 x[N*3]><f32 cov6[N*6]>
[<f32 nodes[A*3]><f32 nodeq[A*2]>] (4-aligned header — JS Float32Array offsets)."""
from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])      # upper-triangle index of a 3x3

_HDR_KEYS = ("loss", "d_vol", "d_render", "d_dt", "d_fill", "lambda", "kin",
             "g_cos", "g_share", "g_phys_norm", "g_rend_norm",
             "v_absmax", "move", "grad_norm", "dfc_absmax", "accepted", "rejected")


def pack_state(seq: int, rec: dict, x: np.ndarray, cov: np.ndarray,
               nodes: np.ndarray | None = None, nodeq: np.ndarray | None = None) -> bytes:
    hdr = {"seq": seq, "n": int(len(x)), "a": 0 if nodes is None else int(len(nodes)),
           "phase": rec.get("phase", "commit"), "sweep": rec.get("sweep"),
           "commit": rec.get("animation", -1) + 1, "run": rec.get("run", 0),
           "E": rec.get("loss"), "lam": rec.get("lambda"),
           "gnorm": rec.get("grad_norm"), "dfc": rec.get("dfc_absmax"),
           "acc": rec.get("accepted"), "rej": rec.get("rejected"),
           "Jmin": rec.get("Jmin_traj", rec.get("Jmin"))}
    for k in ("d_vol", "d_render", "d_dt", "d_fill", "kin", "move",
              "g_cos", "g_share", "g_phys_norm", "g_rend_norm", "v_absmax"):
        hdr[k] = rec.get(k)
    hj = json.dumps(hdr).encode("utf-8")
    hj += b" " * (-len(hj) % 4)
    cov6 = cov[:, _TRI[0], _TRI[1]]
    out = (struct.pack("<I", len(hj)) + hj
           + np.ascontiguousarray(x, "<f4").tobytes()
           + np.ascontiguousarray(cov6, "<f4").tobytes())
    if nodes is not None and len(nodes):
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
    act = np.nonzero(m > 0.5)[0]
    i, j, k = act // (res * res), (act // res) % res, act % res
    nodes = np.stack([i, j, k], 1).astype(np.float32) * ldx + gmin
    nodeq = np.stack([(mv[act] / m[act]), m[act]], 1).astype(np.float32)
    return nodes, nodeq


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = b""
        self.meta = b"{}"
        self.target = b""
        self.history = []
        self.restart = threading.Event()

    def publish(self, blob, hdr=None):
        with self.lock:
            self.state = blob
            if hdr is not None:
                self.history.append(hdr)
                if len(self.history) > 800:
                    self.history.pop(0)

    def snap(self):
        with self.lock:
            return self.state

    def hist_json(self):
        with self.lock:
            return json.dumps(self.history).encode()


def make_handler(hub: Hub, page_dir: Path):
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
                self._send((page_dir / "live.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif self.path.startswith("/quad"):
                self._send((page_dir / "quad.html").read_bytes(),
                           "text/html; charset=utf-8")
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


class LiveServer:
    """One HTTP viewer per process; call begin_run() per arm to get run callbacks."""

    def __init__(self, port: int):
        self.hub = Hub()
        self.port = int(port)
        page_dir = Path(__file__).resolve().parent
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port),
                                         make_handler(self.hub, page_dir))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.seq = 0
        self.run_i = -1
        print(f"[live] serving http://localhost:{self.port}  "
              f"(/quad for the 4-GPU dashboard)", flush=True)

    def begin_run(self, name: str, src: np.ndarray, tgt: np.ndarray, prm, cfg,
                  sigma0: float):
        """Reset per-arm state; returns (on_commit, on_iter) for run_pipeline."""
        from ..render.covariance import cov_from_F
        self.run_i += 1
        run_i = self.run_i
        N = len(tgt)
        extent = float(np.abs(tgt).max()) * 1.25
        gmin = np.asarray(prm.grid_min, np.float32)
        dmax = gmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
        ldx = float((dmax - gmin).max() / cfg.loss_res)
        eye = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        with self.hub.lock:
            self.hub.history = []
        self.hub.meta = json.dumps({
            "n": int(len(src)), "extent": extent, "sigma0": sigma0, "arm": name,
            "T": cfg.T, "iters": cfg.iters, "animations": cfg.animations,
            "pipeline": name}).encode()
        self.hub.target = pack_state(0, {"animation": -1}, tgt, cov_from_F(eye, sigma0))
        self.hub.publish(pack_state(0, {"animation": -1, "run": run_i}, src,
                                    cov_from_F(eye, sigma0)))

        def on_iter(it, xT, FT, tele):
            self.seq += 1
            r = {"animation": -1, "phase": "iter", "sweep": it, "run": run_i, **tele}
            self.hub.publish(pack_state(self.seq, r, xT, cov_from_F(FT, sigma0)))

        def on_commit(a, x, F, v, rec):
            self.seq += 1
            nodes, nodeq = grid_fields(x, v, gmin, ldx, cfg.loss_res)
            r = dict(rec)
            r["run"] = run_i
            r["phase"] = "commit"
            hdr = {"run": run_i, "commit": a + 1, "E": rec.get("loss"),
                   "Jmin": rec.get("Jmin_traj", rec.get("Jmin"))}
            for k in ("d_vol", "d_render", "d_dt", "d_fill", "kin", "move",
                      "g_cos", "g_share", "g_phys_norm", "g_rend_norm"):
                hdr[k] = rec.get(k)
            hdr["lam"] = rec.get("lambda")
            self.hub.publish(pack_state(self.seq, r, x, cov_from_F(F, sigma0),
                                        nodes, nodeq), hdr)

        return on_commit, on_iter
