"""Live-viewer server, embeddable in any run (pipeline_run.py --live_port or --live_dir):
streams accepted iterations + commits over stdlib HTTP for live.html, and serves /quad — the
2x2 dashboard that embeds four ports (one per GPU) so a 4-GPU batch is watchable from one
page. The packet encoding (the binary /state protocol) lives in physmorph.viewer.packets.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .packets import (_HDR_KEYS, _TRI, _array, _json_value, _telemetry_header,  # noqa: F401
                      grid_fields, pack_state, particle_fields)


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
                self.history.append({key: _json_value(value) for key, value in hdr.items()})
                if len(self.history) > 800:
                    self.history.pop(0)

    def snap(self):
        with self.lock:
            return self.state

    def hist_json(self):
        with self.lock:
            return json.dumps(self.history, allow_nan=False).encode()


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
    """One viewer per process; call begin_run() per arm to get run callbacks.

    ``LiveServer(port)`` serves the in-memory Hub over HTTP (legacy, dies with the run).
    ``LiveServer.to_dir(run_dir)`` opens no port: a FileHub persists every packet for the
    standalone scripts/viewer_serve.py.  Both may be combined (``port`` + ``hub``).
    begin_run only touches self.hub / self.seq / self.run_i, so either hub works.
    """

    def __init__(self, port: int | None = None, hub=None):
        self.hub = Hub() if hub is None else hub
        self.port = None if port is None else int(port)
        self.httpd = None
        self.seq = 0
        self.run_i = -1
        if self.port is not None:
            page_dir = Path(__file__).resolve().parent
            self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port),
                                             make_handler(self.hub, page_dir))
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
            print(f"[live] serving http://localhost:{self.port}  "
                  f"(/quad for the 4-GPU dashboard)", flush=True)

    @classmethod
    def to_dir(cls, run_dir, **filehub_kw) -> "LiveServer":
        """Port-less LiveServer whose hub writes ``run_dir`` (see filehub.py)."""
        from .filehub import FileHub
        live = cls(None, hub=FileHub(run_dir, **filehub_kw))
        print(f"[live] writing packets to {Path(run_dir)}  "
              f"(serve with scripts/viewer_serve.py --root <parent>)", flush=True)
        return live

    def begin_run(self, name: str, src: np.ndarray, tgt: np.ndarray, prm, cfg,
                  sigma0: float):
        """Reset per-arm state; returns (on_commit, on_iter) for run_pipeline."""
        from ..render.covariance import cov_from_F
        cov_sat = float(getattr(cfg, "gauss_cov_sat", 0.0) or 0.0)   # rendered saturation
        src = _array("src", src, (None, 3))
        tgt = _array("tgt", tgt, (None, 3))
        if not len(src) or not len(tgt):
            raise ValueError("live viewer requires non-empty source and target clouds")
        if not np.isfinite(src).all() or not np.isfinite(tgt).all():
            raise ValueError("live viewer source and target clouds must be finite")
        self.run_i += 1
        run_i = self.run_i
        self.seq += 1
        extent = max(float(np.abs(tgt).max()) * 1.25, 1e-6)
        gmin = np.asarray(prm.grid_min, np.float32)
        dmax = gmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
        ldx = float((dmax - gmin).max() / cfg.loss_res)
        eye_src = np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))
        eye_tgt = np.tile(np.eye(3, dtype=np.float32), (len(tgt), 1, 1))
        with self.hub.lock:
            self.hub.history = []
        from scipy.spatial import cKDTree
        tgt_tree = cKDTree(tgt)
        nn_sp = (float(np.median(tgt_tree.query(tgt, k=2, workers=-1)[0][:, 1]))
                 if len(tgt) > 1 else 0.0)
        src_rw = np.ones(len(src), np.float32)
        tgt_rw = np.ones(len(tgt), np.float32)
        # surface-only rendering is a viewer option of the earlier configs (absent: off)
        surface_only = bool(getattr(cfg, "render_surface_only", False))
        surface_fraction = float(getattr(cfg, "surface_grad_frac", 0.0))
        if surface_only and surface_fraction <= 0:
            raise ValueError("render_surface_only requires surface_grad_frac > 0")
        if surface_fraction > 0:
            if len(src) < 2 or len(tgt) < 2:
                raise ValueError("surface weights require at least two points per cloud")
            from ..render.surface_weights import surface_weights
            k_sw = int(getattr(cfg, "surface_grad_k", 24))
            floor_sw = float(getattr(cfg, "surface_grad_floor", 0.05))
            src_rw = surface_weights(src, k_sw, surface_fraction, floor_sw)
            tgt_rw = surface_weights(tgt, k_sw, surface_fraction, floor_sw)
        if surface_only:
            from ..render.covariance import sigma0_from_nn
            src_rw = (src_rw > 0.5).astype(np.float32)
            tgt_rw = (tgt_rw > 0.5).astype(np.float32)
            if src_rw.sum() < 1 or tgt_rw.sum() < 2:
                raise ValueError("surface-only viewer requires source/target surface samples")
            sigma0 = sigma0_from_nn(tgt[tgt_rw > 0.5], float(getattr(cfg, "gauss_sigma_scale", 1.0)))
        support = None
        if surface_only:
            # Representation validity only: never edit MPM state or consult the target.
            from ..render.support import MaterialSupport
            support = MaterialSupport.from_rest(src, 8)

        def source_render_weight(x):
            return (src_rw if support is None
                    else np.ascontiguousarray(src_rw * support.opacity(x), np.float32))
        sigma0 = float(sigma0)
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("live viewer sigma0 must be finite and positive")
        child_count = int(getattr(cfg, "gauss_children", 1))
        child_scale = (float(getattr(cfg, "gauss_child_sigma_scale", 0.55))
                       if child_count > 1 else 1.0)
        child_offset_scale = float(getattr(cfg, "gauss_child_offset_scale", 0.35))
        child_k = int(getattr(cfg, "gauss_child_k", 16))
        if child_count < 1 or child_count > 4:
            raise ValueError("live viewer gauss_children must be in [1,4]")
        if not np.isfinite(child_scale) or not 0 < child_scale <= 1:
            raise ValueError("live viewer gauss_child_sigma_scale must be in (0,1]")
        src_mask = ((src_rw > 0.5) if surface_only
                    else np.ones(len(src), dtype=bool))
        tgt_mask = ((tgt_rw > 0.5) if surface_only
                    else np.ones(len(tgt), dtype=bool))
        src_offsets = tgt_offsets = None
        if child_count > 1:
            from ..render.children import tangent_child_offsets
            src_offsets = tangent_child_offsets(src, src_mask, sigma0, child_count,
                                                child_offset_scale, child_k)
            tgt_offsets = tangent_child_offsets(tgt, tgt_mask, sigma0, child_count,
                                                child_offset_scale, child_k)

        def render_payload(x, F, offsets, mask, parent_weight):
            """Viewer-only expansion; parent state remains untouched in the packet."""
            if offsets is None:
                return {}
            from ..render.children import expand_children_numpy
            child_x, child_F = expand_children_numpy(x, F, offsets, mask)
            return {
                "render_x": child_x,
                "render_cov": cov_from_F(child_F, sigma0 * child_scale, sat=cov_sat),
                "render_opacity": np.repeat(np.asarray(parent_weight)[mask], child_count),
            }

        initial_src_weight = source_render_weight(src)
        src_render = render_payload(src, eye_src, src_offsets, src_mask,
                                    initial_src_weight)
        tgt_render = render_payload(tgt, eye_tgt, tgt_offsets, tgt_mask, tgt_rw)
        self.hub.meta = json.dumps({
            "n": int(len(src)), "target_n": int(len(tgt)), "run": run_i,
            "extent": extent, "sigma0": sigma0, "arm": name,
            "T": cfg.T, "iters": cfg.iters, "animations": cfg.animations,
            "nn_sp": nn_sp, "pipeline": name,
            "surface_only": bool(surface_only),
            "surface_count": int((src_rw > 0.5).sum()),
            "target_surface_count": int((tgt_rw > 0.5).sum()),
            "gauss_children": child_count,
            "gauss_child_sigma_scale": child_scale,
            "render_primitive_count": int(src_mask.sum()) * child_count,
            "target_render_primitive_count": int(tgt_mask.sum()) * child_count,
            "canvas_representation": ("massless tangent children: x+F*delta, sigma_child^2*F*F^T"
                                      if child_count > 1 else "parent Gaussian"),
            "render_support": ("target-free frozen material 8-NN opacity"
                               if support is not None else "off"),
            "gradient_snapshot_semantics": ("iteration=that candidate endpoint; accepted commit="
                                            "last iterate at the committed coordinates; rollback/null="
                                            "most recent accepted snapshot at the restored coordinates, "
                                            "or explicitly cleared"),
            "grid_source": "committed particle state, CIC diagnostic",
            "grid_fields": ["mean |v|", "mass", "|momentum|", "specific kinetic", "J", "strain"],
            "particle_fields": ["speed", "J", "condition(F)", "target distance"]},
            allow_nan=False).encode()
        self.hub.target = pack_state(self.seq, {"animation": -1, "phase": "target",
                                               "run": run_i,
                                               "gradient_snapshot": "not_applicable_target"}, tgt,
                                     cov_from_F(eye_tgt, sigma0, sat=cov_sat),
                                     render_weight=tgt_rw, **tgt_render)
        self.hub.publish(pack_state(self.seq, {"animation": -1, "run": run_i,
                                               "gradient_snapshot": "cleared_initial_state"}, src,
                                    cov_from_F(eye_src, sigma0, sat=cov_sat),
                                    render_weight=initial_src_weight, **src_render))

        next_animation = 0
        pending_grad = None
        accepted_grad = None

        def on_iter(it, xT, FT, tele):
            nonlocal next_animation, pending_grad
            self.seq += 1
            tele = dict(tele)
            gp = tele.pop("_grad_phys", None)
            gr = tele.pop("_grad_render", None)
            r = {"animation": next_animation, "phase": "iter", "sweep": it,
                 "run": run_i, **tele}
            pending_grad = {
                "animation": int(next_animation),
                "x": np.asarray(xT, np.float32).copy(),
                "gp": None if gp is None else np.asarray(gp, np.float32).copy(),
                "gr": None if gr is None else np.asarray(gr, np.float32).copy(),
            }
            r["gradient_snapshot"] = "current_window_candidate"
            r["gradient_snapshot_commit"] = int(next_animation) + 1
            dynamic_rw = source_render_weight(xT)
            r["support_faded"] = int(((src_rw > 0.5) & (dynamic_rw < 0.5)).sum())
            child_render = render_payload(xT, FT, src_offsets, src_mask, dynamic_rw)
            self.hub.publish(pack_state(self.seq, r, xT, cov_from_F(FT, sigma0, sat=cov_sat),
                                        grad_phys=gp, grad_render=gr,
                                        render_weight=dynamic_rw, **child_render),
                             _telemetry_header(self.seq, r))

        def on_commit(a, x, F, v, rec):
            nonlocal next_animation, pending_grad, accepted_grad
            self.seq += 1
            nodes, nodeq = grid_fields(x, v, gmin, ldx, cfg.loss_res, F)
            dt_pp = np.full(len(x), np.nan, np.float32)
            finite_x = np.isfinite(x).all(1)
            if finite_x.any():
                dt_pp[finite_x] = tgt_tree.query(x[finite_x], workers=-1)[0].astype(np.float32)
            pq = particle_fields(v, F, dt_pp)
            r = dict(rec)
            r["run"] = run_i
            r["phase"] = "commit"
            rejected = (rec.get("outer_accepted") == 0
                        or bool(rec.get("outer_rejected"))
                        or bool(rec.get("null_commit")))
            commit_gp = commit_gr = None
            if rejected:
                restored_match = (accepted_grad is not None
                                  and np.asarray(x).shape == accepted_grad["x"].shape
                                  and np.isfinite(x).all()
                                  and np.allclose(x, accepted_grad["x"], rtol=1e-6,
                                                  atol=1e-7))
                if restored_match:
                    commit_gp, commit_gr = accepted_grad["gp"], accepted_grad["gr"]
                    r["gradient_snapshot"] = "last_accepted_rollback"
                    r["gradient_snapshot_commit"] = accepted_grad["commit"]
                else:
                    r["gradient_snapshot"] = "cleared_no_matching_accepted_snapshot"
                    r["gradient_snapshot_commit"] = None
            else:
                candidate_match = (pending_grad is not None
                                   and pending_grad["animation"] == int(a)
                                   and (pending_grad["gp"] is not None
                                        or pending_grad["gr"] is not None)
                                   and np.asarray(x).shape == pending_grad["x"].shape
                                   and np.isfinite(x).all()
                                   and np.allclose(x, pending_grad["x"], rtol=1e-6,
                                                   atol=1e-7))
                if candidate_match:
                    commit_gp, commit_gr = pending_grad["gp"], pending_grad["gr"]
                    accepted_grad = {
                        "commit": int(a) + 1,
                        "x": np.asarray(x, np.float32).copy(),
                        "gp": commit_gp,
                        "gr": commit_gr,
                    }
                    r["gradient_snapshot"] = "accepted_current_window"
                    r["gradient_snapshot_commit"] = int(a) + 1
                else:
                    accepted_grad = None
                    r["gradient_snapshot"] = "cleared_no_matching_iter_snapshot"
                    r["gradient_snapshot_commit"] = None
            pending_grad = None
            dynamic_rw = source_render_weight(x)
            visible = ((dynamic_rw > 0.5) if surface_only
                       else np.ones(len(src_rw), dtype=bool))
            r["support_faded"] = int(((src_rw > 0.5) & ~visible).sum())
            r["floater_frac"] = (float((dt_pp[visible] > 2.0 * nn_sp).mean())
                                  if visible.any() else None)
            next_animation = int(a) + 1
            child_render = render_payload(x, F, src_offsets, src_mask, dynamic_rw)
            self.hub.publish(pack_state(self.seq, r, x, cov_from_F(F, sigma0, sat=cov_sat),
                                        nodes, nodeq, dt_pp, particleq=pq,
                                        grad_phys=commit_gp, grad_render=commit_gr,
                                        render_weight=dynamic_rw, **child_render),
                             _telemetry_header(self.seq, r))

        return on_commit, on_iter
