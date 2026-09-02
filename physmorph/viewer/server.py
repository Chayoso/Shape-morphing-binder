"""Live-viewer server, embeddable in ANY run (live_viewer.py CLI or pipeline_run.py
--live_port): streams accepted iterations + commits over stdlib HTTP for live.html,
and serves /quad — the 2x2 dashboard that embeds four ports (one per GPU) so a full
4-GPU batch is watchable in real time from one page.

Binary /state protocol: ``<u32 hlen><json hdr>`` followed by little-endian
float32 arrays.  v2 is ``x[N,3]``, ``cov6[N,6]`` then its optional diagnostics.
v3 inserts the objective-visible render representation immediately after those
parent arrays: ``render_x[R,3]``, ``render_cov6[R,6]``, ``render_opacity[R]``.
Physics/grid/gradient arrays remain parent-sized.  The padded JSON header and
payload are both 4-byte aligned for JS typed arrays.
"""
from __future__ import annotations

import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_TRI = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])      # upper-triangle index of a 3x3

_HDR_KEYS = ("loss", "d_vol", "d_render", "d_dt", "d_fill", "lambda", "kin",
             "g_raw_cos", "g_cos", "g_share", "g_phys_norm", "g_rend_norm",
             "render_work", "render_work_x", "render_work_F",
             "phys_work", "phys_work_x", "phys_work_F", "phys_work_v",
             "step_norm", "predicted_decrease",
             "v_absmax", "v_mean", "move", "grad_norm", "dfc_absmax",
             "accepted", "rejected", "outer_merit", "outer_gain", "reversal_cos",
             "outer_accepted", "outer_gate_latched", "gauss_condition_p95", "gauss_condition_max",
             "gauss_radius_over_spacing_p95", "gauss_radius_over_spacing_max",
             "floater_frac", "support_faded", "gradient_snapshot",
             "gradient_snapshot_commit")


def _json_value(value):
    """Convert scalar telemetry to strict JSON; non-finite values become ``null``."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _telemetry_header(seq: int, rec: dict) -> dict:
    hdr = {
        "seq": int(seq), "phase": rec.get("phase", "commit"),
        "sweep": rec.get("sweep"), "commit": int(rec.get("animation", -1)) + 1,
        "run": rec.get("run", 0), "E": rec.get("loss"),
        "lam": rec.get("lambda"), "gnorm": rec.get("grad_norm"),
        "dfc": rec.get("dfc_absmax"), "acc": rec.get("accepted"),
        "rej": rec.get("rejected"),
        "Jmin": rec.get("Jmin_traj", rec.get("Jmin")),
    }
    for key in _HDR_KEYS:
        hdr[key] = rec.get(key)
    return {key: _json_value(value) for key, value in hdr.items()}


def _array(name: str, value, shape: tuple[int | None, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != len(shape) or any(want is not None and got != want
                                     for got, want in zip(arr.shape, shape)):
        expected = "x".join("*" if n is None else str(n) for n in shape)
        raise ValueError(f"{name} must have shape ({expected}), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype="<f4")


def pack_state(seq: int, rec: dict, x: np.ndarray, cov: np.ndarray,
               nodes: np.ndarray | None = None, nodeq: np.ndarray | None = None,
               dt_pp: np.ndarray | None = None, particleq: np.ndarray | None = None,
               grad_phys: np.ndarray | None = None,
               grad_render: np.ndarray | None = None,
               render_weight: np.ndarray | None = None,
               render_x: np.ndarray | None = None,
               render_cov: np.ndarray | None = None,
               render_opacity: np.ndarray | None = None) -> bytes:
    x = _array("x", x, (None, 3))
    n = len(x)
    cov = _array("cov", cov, (n, 3, 3))
    if (nodes is None) != (nodeq is None):
        raise ValueError("nodes and nodeq must either both be present or both be absent")
    if nodes is not None:
        nodes = _array("nodes", nodes, (None, 3))
        nodeq = _array("nodeq", nodeq, (len(nodes), None))
    if dt_pp is not None:
        dt_pp = _array("dt_pp", dt_pp, (n,))
    if particleq is not None:
        particleq = _array("particleq", particleq, (n, None))
    if grad_phys is not None:
        grad_phys = _array("grad_phys", grad_phys, (n, 3))
    if grad_render is not None:
        grad_render = _array("grad_render", grad_render, (n, 3))
    if render_weight is not None:
        render_weight = _array("render_weight", render_weight, (n,))
    render_values = (render_x, render_cov, render_opacity)
    if any(v is not None for v in render_values) and not all(v is not None
                                                              for v in render_values):
        raise ValueError("render_x, render_cov, and render_opacity must be provided together")
    if render_x is not None:
        render_x = _array("render_x", render_x, (None, 3))
        render_n = len(render_x)
        render_cov = _array("render_cov", render_cov, (render_n, 3, 3))
        render_opacity = _array("render_opacity", render_opacity, (render_n,))
    else:
        render_n = 0

    arrays = [x, cov[:, _TRI[0], _TRI[1]]]
    if render_x is not None:
        arrays += [render_x, render_cov[:, _TRI[0], _TRI[1]], render_opacity]
    arrays += ([] if nodes is None else [nodes, nodeq])
    arrays += [a for a in (dt_pp, particleq, grad_phys, grad_render, render_weight)
               if a is not None]
    hdr = _telemetry_header(seq, rec)
    hdr.update({
        "protocol": 3 if render_x is not None else 2, "n": n,
        "r": render_n, "render_primitives": render_x is not None,
        "a": 0 if nodes is None else int(len(nodes)),
        "q": 0 if nodeq is None else int(nodeq.shape[1]),
        "pq": 0 if particleq is None else int(particleq.shape[1]),
        "dt_pp": dt_pp is not None,
        "grad_phys": grad_phys is not None, "grad_render": grad_render is not None,
        "render_weight": render_weight is not None,
        "payload_floats": int(sum(a.size for a in arrays)),
        "nonfinite": {
            name: int(np.size(a) - np.isfinite(a).sum())
            for name, a in (("x", x), ("cov", cov), ("nodes", nodes),
                            ("nodeq", nodeq), ("dt", dt_pp),
                            ("particleq", particleq), ("grad_phys", grad_phys),
                            ("grad_render", grad_render),
                            ("render_weight", render_weight),
                            ("render_x", render_x), ("render_cov", render_cov),
                            ("render_opacity", render_opacity))
            if a is not None
        },
    })
    hj = json.dumps(hdr, allow_nan=False).encode("utf-8")
    hj += b" " * (-len(hj) % 4)
    return struct.pack("<I", len(hj)) + hj + b"".join(a.tobytes() for a in arrays)


def grid_fields(x, v, gmin, ldx, res, F=None):
    """CIC diagnostics: mean speed, mass, momentum, kinetic, J and strain."""
    x = _array("x", x, (None, 3))
    v = _array("v", v, (len(x), 3))
    if F is None:
        F = np.tile(np.eye(3, dtype=np.float32), (len(x), 1, 1))
    F = _array("F", F, (len(x), 3, 3))
    gmin = _array("gmin", gmin, (3,))
    if int(res) <= 0 or not np.isfinite(ldx) or float(ldx) <= 0:
        raise ValueError("res and ldx must be positive")
    valid = (np.isfinite(x).all(1) & np.isfinite(v).all(1)
             & np.isfinite(F).all(axis=(1, 2)))
    x, v, F = x[valid], v[valid], F[valid]
    rel = (x - gmin) / ldx
    base = np.floor(rel).astype(np.int64)
    frac = rel - base
    m = np.zeros(res ** 3, np.float64)
    mv = np.zeros(res ** 3, np.float64)
    mom = np.zeros((res ** 3, 3), np.float64)
    ke = np.zeros(res ** 3, np.float64)
    j_acc = np.zeros(res ** 3, np.float64)
    s_acc = np.zeros(res ** 3, np.float64)
    Jp = np.linalg.det(F)
    strainp = np.linalg.norm(np.swapaxes(F, 1, 2) @ F - np.eye(3), axis=(1, 2))
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
                np.add.at(mom, idx, w[:, None] * v)
                np.add.at(ke, idx, w * 0.5 * sp * sp)
                np.add.at(j_acc, idx, w * Jp)
                np.add.at(s_acc, idx, w * strainp)
    # Every positive-weight CIC node is real diagnostic data.  The old 0.5 cutoff
    # dropped all eight nodes of a particle centred in a cell (weight 0.125 each).
    act = np.nonzero(m > 1e-12)[0]
    i, j, k = act // (res * res), (act // res) % res, act % res
    nodes = np.stack([i, j, k], 1).astype(np.float32) * ldx + gmin
    nodeq = np.stack([mv[act] / m[act], m[act],
                      np.linalg.norm(mom[act], axis=1), ke[act] / m[act],
                      j_acc[act] / m[act], s_acc[act] / m[act]], 1).astype(np.float32)
    return nodes, nodeq


def particle_fields(v, F, dt_pp):
    """Raw-state particle quantities: speed, J, condition(F), target distance."""
    v = _array("v", v, (None, 3))
    F = _array("F", F, (len(v), 3, 3))
    dt_pp = _array("dt_pp", dt_pp, (len(v),))
    out = np.full((len(v), 4), np.nan, np.float32)
    good_v = np.isfinite(v).all(1)
    out[good_v, 0] = np.linalg.norm(v[good_v], axis=1)
    good_F = np.isfinite(F).all(axis=(1, 2))
    if good_F.any():
        Fg = F[good_F]
        sv = np.linalg.svd(Fg, compute_uv=False)
        out[good_F, 1] = np.linalg.det(Fg)
        out[good_F, 2] = sv[:, 0] / np.maximum(sv[:, -1], 1e-8)
    out[:, 3] = dt_pp
    return out


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
        surface_fraction = float(getattr(cfg, "surface_grad_frac", 0.0))
        if cfg.render_surface_only and surface_fraction <= 0:
            raise ValueError("render_surface_only requires surface_grad_frac > 0")
        if surface_fraction > 0:
            if len(src) < 2 or len(tgt) < 2:
                raise ValueError("surface weights require at least two points per cloud")
            from ..pipeline.runner import _surface_weights
            src_rw = _surface_weights(src, cfg.surface_grad_k, surface_fraction,
                                      cfg.surface_grad_floor)
            tgt_rw = _surface_weights(tgt, cfg.surface_grad_k, surface_fraction,
                                      cfg.surface_grad_floor)
        if cfg.render_surface_only:
            from ..render.covariance import sigma0_from_nn
            src_rw = (src_rw > 0.5).astype(np.float32)
            tgt_rw = (tgt_rw > 0.5).astype(np.float32)
            if src_rw.sum() < 1 or tgt_rw.sum() < 2:
                raise ValueError("surface-only viewer requires source/target surface samples")
            sigma0 = sigma0_from_nn(tgt[tgt_rw > 0.5], cfg.gauss_sigma_scale)
        support = None
        if cfg.render_surface_only:
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
        src_mask = ((src_rw > 0.5) if cfg.render_surface_only
                    else np.ones(len(src), dtype=bool))
        tgt_mask = ((tgt_rw > 0.5) if cfg.render_surface_only
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
                "render_cov": cov_from_F(child_F, sigma0 * child_scale),
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
            "surface_only": bool(cfg.render_surface_only),
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
                                     cov_from_F(eye_tgt, sigma0),
                                     render_weight=tgt_rw, **tgt_render)
        self.hub.publish(pack_state(self.seq, {"animation": -1, "run": run_i,
                                               "gradient_snapshot": "cleared_initial_state"}, src,
                                    cov_from_F(eye_src, sigma0),
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
            self.hub.publish(pack_state(self.seq, r, xT, cov_from_F(FT, sigma0),
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
            visible = ((dynamic_rw > 0.5) if cfg.render_surface_only
                       else np.ones(len(src_rw), dtype=bool))
            r["support_faded"] = int(((src_rw > 0.5) & ~visible).sum())
            r["floater_frac"] = (float((dt_pp[visible] > 2.0 * nn_sp).mean())
                                  if visible.any() else None)
            next_animation = int(a) + 1
            child_render = render_payload(x, F, src_offsets, src_mask, dynamic_rw)
            self.hub.publish(pack_state(self.seq, r, x, cov_from_F(F, sigma0),
                                        nodes, nodeq, dt_pp, particleq=pq,
                                        grad_phys=commit_gp, grad_render=commit_gr,
                                        render_weight=dynamic_rw, **child_render),
                             _telemetry_header(self.seq, r))

        return on_commit, on_iter
