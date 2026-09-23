import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physmorph.viewer.server import Hub, LiveServer, grid_fields, pack_state, particle_fields
from physmorph.pipeline.optimizer import _linearized_work


def _decode(blob):
    hlen = struct.unpack_from("<I", blob)[0]
    hdr = json.loads(blob[4:4 + hlen])
    off = 4 + hlen

    def take(count):
        nonlocal off
        out = np.frombuffer(blob, "<f4", count=count, offset=off).copy()
        off += count * 4
        return out

    n, r, a, q, pq = hdr["n"], hdr.get("r", 0), hdr["a"], hdr["q"], hdr["pq"]
    arrays = {"x": take(n * 3).reshape(n, 3), "cov6": take(n * 6).reshape(n, 6)}
    if hdr.get("render_primitives"):
        arrays["render_x"] = take(r * 3).reshape(r, 3)
        arrays["render_cov6"] = take(r * 6).reshape(r, 6)
        arrays["render_opacity"] = take(r)
    if a:
        arrays["nodes"] = take(a * 3).reshape(a, 3)
        arrays["nodeq"] = take(a * q).reshape(a, q)
    if hdr["dt_pp"]:
        arrays["dt"] = take(n)
    if pq:
        arrays["particleq"] = take(n * pq).reshape(n, pq)
    if hdr["grad_phys"]:
        arrays["gp"] = take(n * 3).reshape(n, 3)
    if hdr["grad_render"]:
        arrays["gr"] = take(n * 3).reshape(n, 3)
    if hdr["render_weight"]:
        arrays["rw"] = take(n)
    assert off == len(blob)
    return hdr, arrays


def test_diagnostic_binary_layout_and_flags():
    n, a = 4, 3
    x = np.zeros((n, 3), np.float32)
    cov = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    nodes = np.zeros((a, 3), np.float32)
    nodeq = np.zeros((a, 6), np.float32)
    dt = np.arange(n, dtype=np.float32)
    pq = np.zeros((n, 4), np.float32)
    gp = np.ones((n, 3), np.float32)
    gr = -gp
    rw = np.array([1, 0, 1, 0], np.float32)
    blob = pack_state(7, {"loss": 2.0}, x, cov, nodes, nodeq, dt, pq, gp, gr,
                      render_weight=rw)
    hdr, arrays = _decode(blob)
    assert hdr["protocol"] == 2
    assert (hdr["q"], hdr["pq"], hdr["grad_phys"], hdr["grad_render"],
            hdr["render_weight"]) == (6, 4, True, True, True)
    assert hdr["payload_floats"] == sum(a.size for a in arrays.values())
    assert np.array_equal(arrays["dt"], dt)
    assert np.array_equal(arrays["gp"], gp)
    assert np.array_equal(arrays["gr"], gr)
    assert np.array_equal(arrays["rw"], rw)
    hlen = struct.unpack_from("<I", blob)[0]
    expected = (4 + hlen + n * 36 + a * (3 + 6) * 4 + n * 4 + n * 4 * 4
                + n * 3 * 4 * 2 + n * 4)
    assert len(blob) == expected


def test_v3_render_primitive_layout_is_separate_from_parent_diagnostics():
    n, r = 3, 5
    x = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    cov = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    render_x = np.arange(r * 3, dtype=np.float32).reshape(r, 3) / 10
    render_cov = np.tile(2 * np.eye(3, dtype=np.float32), (r, 1, 1))
    render_opacity = np.linspace(0.2, 1.0, r, dtype=np.float32)
    gp = np.ones((n, 3), np.float32)
    blob = pack_state(9, {}, x, cov, grad_phys=gp,
                      render_x=render_x, render_cov=render_cov,
                      render_opacity=render_opacity)
    hdr, arrays = _decode(blob)
    assert hdr["protocol"] == 3 and hdr["render_primitives"] is True
    assert (hdr["n"], hdr["r"]) == (n, r)
    assert arrays["gp"].shape == (n, 3)
    assert np.array_equal(arrays["render_x"], render_x)
    assert np.array_equal(arrays["render_opacity"], render_opacity)
    with pytest.raises(ValueError, match="must be provided together"):
        pack_state(9, {}, x, cov, render_x=render_x)


def test_grid_and_particle_fields_are_finite():
    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], np.float32)
    v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], np.float32)
    F = np.stack([np.eye(3), np.diag([2.0, 1.0, 0.5])]).astype(np.float32)
    nodes, q = grid_fields(x, v, np.array([-1.0] * 3, np.float32), 0.25, 16, F)
    pq = particle_fields(v, F, np.array([0.0, 0.2], np.float32))
    assert nodes.shape[1] == 3 and q.shape == (len(nodes), 6)
    assert pq.shape == (2, 4)
    assert np.isfinite(q).all() and np.isfinite(pq).all()
    assert pq[1, 2] == 4.0


def test_grid_keeps_low_weight_nodes_and_handles_empty_or_nonfinite_rows():
    # One particle at a cell centre contributes 0.125 to eight real CIC nodes; none
    # may disappear behind a display threshold.
    gmin = np.zeros(3, np.float32)
    x = np.array([[0.5, 0.5, 0.5], [np.nan, 0.0, 0.0]], np.float32)
    v = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32)
    F = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    nodes, q = grid_fields(x, v, gmin, 1.0, 3, F)
    assert nodes.shape == (8, 3) and q.shape == (8, 6)
    assert q[:, 1].sum() == pytest.approx(1.0)
    empty_nodes, empty_q = grid_fields(np.empty((0, 3), np.float32),
                                       np.empty((0, 3), np.float32), gmin, 1.0, 3)
    assert empty_nodes.shape == (0, 3) and empty_q.shape == (0, 6)


def test_protocol_rejects_shape_mismatch_and_serializes_nonfinite_header_strictly():
    x = np.array([[np.nan, 0.0, 0.0]], np.float32)
    cov = np.eye(3, dtype=np.float32)[None]
    blob = pack_state(1, {"loss": float("nan")}, x, cov)
    hdr, arrays = _decode(blob)
    assert hdr["E"] is None and hdr["loss"] is None
    assert hdr["nonfinite"]["x"] == 1 and np.isnan(arrays["x"][0, 0])
    assert b"NaN" not in blob[:4 + struct.unpack_from("<I", blob)[0]]
    with pytest.raises(ValueError, match="cov must have shape"):
        pack_state(1, {}, np.zeros((2, 3), np.float32), cov)
    with pytest.raises(ValueError, match="nodes and nodeq"):
        pack_state(1, {}, np.zeros((1, 3), np.float32), cov,
                   nodes=np.zeros((1, 3), np.float32))

    empty_blob = pack_state(2, {}, np.empty((0, 3), np.float32),
                            np.empty((0, 3, 3), np.float32))
    empty_hdr, empty_arrays = _decode(empty_blob)
    assert empty_hdr["n"] == 0 and empty_hdr["payload_floats"] == 0
    assert empty_arrays["x"].shape == (0, 3)


def test_particle_fields_preserve_invalid_row_as_diagnostic_nan():
    v = np.array([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], np.float32)
    F = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    F[1, 0, 0] = np.nan
    q = particle_fields(v, F, np.array([0.1, np.nan], np.float32))
    assert np.isfinite(q[0]).all()
    assert np.isnan(q[1]).all()


def test_endpoint_work_includes_deformation_gradient_component():
    import torch
    gx = torch.tensor([[1.0, 0.0, 0.0]])
    gF = torch.full((1, 9), 2.0)
    dx = torch.tensor([[3.0, 0.0, 0.0]])
    dF = torch.ones(1, 9)
    total, parts = _linearized_work((gx, gF), (dx, dF))
    assert parts == [-3.0, -18.0]
    assert total == -21.0


def test_begin_run_supports_different_source_and_target_counts():
    live = LiveServer.__new__(LiveServer)
    live.hub, live.seq, live.run_i = Hub(), 0, -1
    src = np.zeros((3, 3), np.float32)
    tgt = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]], np.float32)
    prm = SimpleNamespace(grid_min=(-1.0, -1.0, -1.0), dx=0.25, nx=8, ny=8, nz=8)
    cfg = SimpleNamespace(loss_res=8, render_surface_only=False, T=2, iters=3,
                          animations=4)
    _, on_iter = live.begin_run("parity", src, tgt, prm, cfg, sigma0=0.1)
    src_hdr, _ = _decode(live.hub.state)
    tgt_hdr, _ = _decode(live.hub.target)
    assert src_hdr["n"] == len(src) and tgt_hdr["n"] == len(tgt)
    assert src_hdr["protocol"] == 2 and tgt_hdr["protocol"] == 2
    assert not src_hdr["render_primitives"] and not tgt_hdr["render_primitives"]
    meta = json.loads(live.hub.meta)
    assert (meta["n"], meta["target_n"]) == (len(src), len(tgt))
    F = np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))
    gp = np.ones_like(src)
    on_iter(2, src, F, {"loss": 3.0, "d_vol": 2.0,
                         "_grad_phys": gp, "_grad_render": -gp})
    history = json.loads(live.hub.hist_json())
    assert len(history) == 1 and history[0]["phase"] == "iter"
    assert history[0]["sweep"] == 2 and history[0]["E"] == 3.0
    iter_hdr, arrays = _decode(live.hub.state)
    assert iter_hdr["commit"] == 1
    assert np.array_equal(arrays["gp"], gp) and np.array_equal(arrays["gr"], -gp)


def test_surface_viewer_fades_only_target_free_unsupported_render_primitive():
    a = np.linspace(-0.2, 0.2, 3, dtype=np.float32)
    src = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    live = LiveServer.__new__(LiveServer)
    live.hub, live.seq, live.run_i = Hub(), 0, -1
    prm = SimpleNamespace(grid_min=(-2.0, -2.0, -2.0), dx=0.5, nx=12, ny=12, nz=12)
    cfg = SimpleNamespace(loss_res=8, render_surface_only=True, T=2, iters=3,
                          animations=4, surface_grad_frac=0.8, surface_grad_k=8,
                          surface_grad_floor=0.05, gauss_sigma_scale=1.0)
    on_commit, _ = live.begin_run("support", src, src.copy(), prm, cfg, sigma0=0.1)
    _, initial = _decode(live.hub.state)
    p = int(np.flatnonzero(initial["rw"] > 0.5)[0])
    x = src.copy(); x[p, 0] += 4.0
    F = np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))
    on_commit(0, x, F, np.zeros_like(x), {"animation": 0})
    hdr, arrays = _decode(live.hub.state)
    assert arrays["rw"][p] < 0.5
    assert hdr["support_faded"] >= 1
    assert hdr["gradient_snapshot"] == "cleared_no_matching_iter_snapshot"
    assert not hdr["grad_phys"] and not hdr["grad_render"]
    assert json.loads(live.hub.meta)["render_support"].startswith("target-free")


def test_live_viewer_streams_actual_children_but_keeps_parent_gradient_count():
    a = np.linspace(-0.2, 0.2, 3, dtype=np.float32)
    src = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    live = LiveServer.__new__(LiveServer)
    live.hub, live.seq, live.run_i = Hub(), 0, -1
    prm = SimpleNamespace(grid_min=(-2.0, -2.0, -2.0), dx=0.5, nx=12, ny=12, nz=12)
    cfg = SimpleNamespace(loss_res=8, render_surface_only=True, T=2, iters=3,
                          animations=4, surface_grad_frac=0.8, surface_grad_k=8,
                          surface_grad_floor=0.05, gauss_sigma_scale=1.0,
                          gauss_children=4, gauss_child_sigma_scale=0.55,
                          gauss_child_offset_scale=0.35, gauss_child_k=8)
    _, on_iter = live.begin_run("children", src, src.copy(), prm, cfg, sigma0=0.1)
    initial_hdr, initial = _decode(live.hub.state)
    target_hdr, target = _decode(live.hub.target)
    meta = json.loads(live.hub.meta)
    active = initial["rw"] > 0.5
    render_n = int(active.sum()) * 4
    assert initial_hdr["protocol"] == 3 and initial_hdr["n"] == len(src)
    assert initial_hdr["r"] == render_n
    assert initial["render_x"].shape == (render_n, 3)
    assert target_hdr["protocol"] == 3
    assert target["render_x"].shape == (meta["target_render_primitive_count"], 3)
    np.testing.assert_allclose(initial["render_opacity"].reshape(-1, 4)[:, 0],
                               initial["rw"][active])

    parent_centers = np.repeat(src[active], 4, axis=0)
    rest_delta = initial["render_x"] - parent_centers
    x = src + np.array([0.03, -0.02, 0.01], np.float32)
    F = np.tile(2 * np.eye(3, dtype=np.float32), (len(src), 1, 1))
    gp = np.ones_like(src)
    on_iter(0, x, F, {"_grad_phys": gp, "_grad_render": -gp})
    hdr, state = _decode(live.hub.state)
    expected = np.repeat(x[active], 4, axis=0) + 2.0 * rest_delta
    np.testing.assert_allclose(state["render_x"], expected, atol=1e-6)
    child_var = (meta["sigma0"] * 0.55 * 2.0) ** 2
    np.testing.assert_allclose(state["render_cov6"][:, (0, 3, 5)], child_var,
                               rtol=1e-5)
    assert state["gp"].shape == (len(src), 3)
    assert hdr["n"] == len(src) and hdr["r"] == render_n
    assert meta["render_primitive_count"] == render_n
    assert meta["canvas_representation"].startswith("massless tangent children")


def test_rejected_commit_reuses_only_the_last_accepted_gradient_snapshot():
    src = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0]], np.float32)
    live = LiveServer.__new__(LiveServer)
    live.hub, live.seq, live.run_i = Hub(), 0, -1
    prm = SimpleNamespace(grid_min=(-1.0, -1.0, -1.0), dx=0.25, nx=8, ny=8, nz=8)
    cfg = SimpleNamespace(loss_res=8, render_surface_only=False, T=2, iters=3,
                          animations=4)
    on_commit, on_iter = live.begin_run("rollback-gradient", src, src.copy(), prm,
                                        cfg, sigma0=0.1)
    F = np.tile(np.eye(3, dtype=np.float32), (len(src), 1, 1))
    v = np.zeros_like(src)

    accepted_x = src + np.array([0.01, 0.0, 0.0], np.float32)
    accepted_gp = np.arange(9, dtype=np.float32).reshape(3, 3) + 1
    accepted_gr = -2.0 * accepted_gp
    on_iter(0, accepted_x, F, {"_grad_phys": accepted_gp,
                               "_grad_render": accepted_gr})
    on_commit(0, accepted_x, F, v, {"animation": 0, "outer_accepted": 1})
    accepted_hdr, accepted_state = _decode(live.hub.state)
    assert accepted_hdr["gradient_snapshot"] == "accepted_current_window"
    assert accepted_hdr["gradient_snapshot_commit"] == 1
    assert np.array_equal(accepted_state["gp"], accepted_gp)
    assert np.array_equal(accepted_state["gr"], accepted_gr)

    rejected_x = accepted_x + np.array([0.3, 0.1, 0.0], np.float32)
    rejected_gp = np.full_like(src, 99.0)
    rejected_gr = np.full_like(src, -77.0)
    on_iter(1, rejected_x, F, {"_grad_phys": rejected_gp,
                               "_grad_render": rejected_gr})
    on_commit(1, accepted_x, F, v, {"animation": 1, "outer_accepted": 0,
                                    "outer_rejected": 1, "null_commit": 1})
    rollback_hdr, rollback_state = _decode(live.hub.state)
    assert np.array_equal(rollback_state["x"], accepted_x)
    assert np.array_equal(rollback_state["gp"], accepted_gp)
    assert np.array_equal(rollback_state["gr"], accepted_gr)
    assert not np.array_equal(rollback_state["gp"], rejected_gp)
    assert rollback_hdr["gradient_snapshot"] == "last_accepted_rollback"
    assert rollback_hdr["gradient_snapshot_commit"] == 1
    assert "restored coordinates" in json.loads(live.hub.meta)[
        "gradient_snapshot_semantics"]


def test_live_page_exposes_every_diagnostic_toggle_and_v3_length_check():
    html = (Path(__file__).parents[1] / "physmorph" / "viewer" / "live.html").read_text()
    for element_id in ("iterations", "splats", "surfaceOnly", "maskOn", "targetOn",
                       "floaters", "particles", "gridOn", "physGrad", "renderGrad"):
        assert f'id="{element_id}"' in html
    assert "h.protocol!==2&&h.protocol!==3" in html
    assert "renderState(target)" in html and "renderState(shown)" in html
    assert "lastGrad" not in html
    assert "h.gradient_snapshot" in html
    assert "rn*(3+6+1)" in html
    assert "payload/header length mismatch" in html
    assert "render_work_F" in html and "phys_work_F" in html and "phys_work_v" in html
