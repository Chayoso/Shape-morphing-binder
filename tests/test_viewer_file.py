"""Decoupled viewer: FileHub persistence, LiveServer.to_dir, viewer_serve HTTP contract,
page contract, viewer_tunnel probe.  CPU only, no GPU; every test well under 2 s."""
import importlib.util
import json
import socket
import struct
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physmorph.viewer.filehub import FileHub
from physmorph.viewer.server import Hub, LiveServer, _telemetry_header, pack_state

ROOT = Path(__file__).resolve().parents[1]


def _decode(blob):
    """Copy of tests/test_viewer_server.py::_decode (kept local: no cross-test import)."""
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


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _packet(seq, phase, n=3):
    x = np.full((n, 3), float(seq), np.float32)
    cov = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    rec = {"animation": 0, "phase": phase, "loss": float(seq)}
    return pack_state(seq, rec, x, cov), _telemetry_header(seq, rec)


def _commit_names(run):
    return sorted(p.name for p in (run / "commits").glob("*.bin"))


def _http(port, path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_filehub_round_trip_and_restart_flag(tmp_path):
    run = tmp_path / "run"
    hub = FileHub(run, max_commits=2)
    hub.meta = b'{"arm": "a"}'
    tgt_blob, _ = _packet(1, "target")
    hub.target = tgt_blob
    assert json.loads((run / "meta.json").read_bytes()) == {"arm": "a"}
    assert (run / "target.bin").read_bytes() == tgt_blob and hub.target == tgt_blob

    it_blob, it_hdr = _packet(1, "iter")
    hub.publish(it_blob, it_hdr)
    assert hub.snap() == it_blob and (run / "state.bin").read_bytes() == it_blob
    assert _commit_names(run) == []                      # iter packets are not stored
    assert [h["seq"] for h in json.loads((run / "history.json").read_bytes())] == [1]
    hub.publish(*_packet(2, "iter"))                     # inside the 0.5 s throttle
    assert [h["seq"] for h in json.loads((run / "history.json").read_bytes())] == [1]

    c_blob, c_hdr = _packet(3, "commit")
    hub.publish(c_blob, c_hdr)                           # commit: stored + history flushed
    assert (run / "state.bin").read_bytes() == c_blob
    assert _commit_names(run) == ["000003.bin"]
    hdr, arrays = _decode((run / "commits" / "000003.bin").read_bytes())
    assert hdr["seq"] == 3 and hdr["phase"] == "commit" and hdr["E"] == 3.0
    assert arrays["x"].shape == (3, 3) and arrays["x"][0, 0] == 3.0
    hist = json.loads((run / "history.json").read_bytes())
    assert [h["seq"] for h in hist] == [1, 2, 3]
    assert hist == json.loads(hub.hist_json())

    # begin_run's initial packet carries no phase -> header phase "commit" -> stored.
    hub.publish(pack_state(4, {"animation": -1}, np.zeros((2, 3), np.float32),
                           np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))))
    assert _commit_names(run) == ["000003.bin", "000004.bin"]
    hub.publish(*_packet(5, "commit"))                   # max_commits=2 prunes the oldest
    assert _commit_names(run) == ["000004.bin", "000005.bin"]
    assert hub.io_errors == 0

    flag = run / "restart.flag"
    assert not hub.restart.is_set()
    flag.touch()
    assert hub.poll_restart() and hub.restart.is_set() and not flag.exists()
    hub.restart.clear()
    flag.touch()
    assert hub.restart.wait(1.0) and not flag.exists()   # hold-mode wait() sees the file
    hub.restart.clear()
    assert not hub.restart.wait(0.05)


def test_to_dir_begin_run_writes_run_files(tmp_path):
    d = tmp_path / "r1"
    live = LiveServer.to_dir(d)
    assert live.httpd is None and live.port is None and isinstance(live.hub, FileHub)
    src = np.zeros((3, 3), np.float32)
    tgt = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
                    [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]], np.float32)
    prm = SimpleNamespace(grid_min=(-1.0, -1.0, -1.0), dx=0.25, nx=8, ny=8, nz=8)
    cfg = SimpleNamespace(loss_res=8, render_surface_only=False, T=2, iters=3,
                          animations=4)
    on_commit, on_iter = live.begin_run("parity", src, tgt, prm, cfg, sigma0=0.1)
    meta = json.loads((d / "meta.json").read_bytes())
    assert meta["arm"] == "parity" and (meta["n"], meta["target_n"]) == (3, 4)
    tgt_hdr, _ = _decode((d / "target.bin").read_bytes())
    st_hdr, _ = _decode((d / "state.bin").read_bytes())
    assert tgt_hdr["phase"] == "target" and tgt_hdr["n"] == 4
    assert st_hdr["n"] == 3 and st_hdr["seq"] == 1
    assert _commit_names(d) == ["000001.bin"]            # initial state is replayable
    F = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    on_iter(0, src, F, {"loss": 1.0})
    it_hdr, _ = _decode((d / "state.bin").read_bytes())
    assert it_hdr["phase"] == "iter" and _commit_names(d) == ["000001.bin"]
    on_commit(0, src, F, np.zeros_like(src), {"animation": 0, "outer_accepted": 1})
    c_hdr, arrays = _decode((d / "state.bin").read_bytes())
    assert c_hdr["phase"] == "commit" and c_hdr["commit"] == 1 and "nodes" in arrays
    assert _commit_names(d) == ["000001.bin", "000003.bin"]
    rows = json.loads((d / "history.json").read_bytes())
    assert [r["phase"] for r in rows] == ["iter", "commit"]


def test_port_mode_still_opens_http_server():
    live = LiveServer(0)
    try:
        assert isinstance(live.hub, Hub) and live.httpd.server_address[1] > 0
        code, body, _ = _http(live.httpd.server_address[1], "/runs")
        assert code == 404                               # legacy server: no run listing
    finally:
        live.httpd.shutdown()
        live.httpd.server_close()


@pytest.fixture
def served(tmp_path):
    vs = _load_script("viewer_serve")
    hub = FileHub(tmp_path / "runA")
    hub.meta = json.dumps({"arm": "render", "n": 3}).encode()
    hub.target = _packet(1, "target")[0]
    hub.publish(*_packet(1, "commit"))
    hub.publish(*_packet(2, "iter"))
    hub.publish(*_packet(3, "commit"))
    (tmp_path / "notarun").mkdir()                       # no meta.json -> not a run
    (tmp_path / "loose.txt").write_text("x")
    httpd = vs.make_server(tmp_path, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield vs, httpd.server_address[1], tmp_path, hub
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_viewer_serve_contract(served):
    vs, port, root, hub = served
    code, body, headers = _http(port, "/runs")
    assert code == 200 and headers.get("Cache-Control") == "no-store"
    runs = json.loads(body)
    assert [r["name"] for r in runs] == ["runA"]
    r = runs[0]
    assert r["arm"] == "render" and r["seq"] == 3 and r["commits"] == 2
    assert r["live"] is True and 0 <= r["age"] < 60
    code, body, _ = _http(port, "/r/runA/meta")
    assert code == 200 and json.loads(body)["arm"] == "render"
    code, body, _ = _http(port, "/r/runA/state")
    assert code == 200 and _decode(body)[0]["seq"] == 3
    code, body, _ = _http(port, "/r/runA/target")
    assert code == 200 and _decode(body)[0]["phase"] == "target"
    code, body, _ = _http(port, "/r/runA/history")
    assert code == 200 and [h["seq"] for h in json.loads(body)] == [1, 2, 3]
    code, body, _ = _http(port, "/r/runA/commits")
    assert code == 200 and json.loads(body) == [1, 3]
    code, body, _ = _http(port, "/r/runA/commit/1")
    assert code == 200 and _decode(body)[0]["seq"] == 1
    assert _http(port, "/r/runA/commit/2")[0] == 503     # iter packets are not stored
    assert _http(port, "/r/runA/commit/x")[0] == 404
    for page, marker in (("/", "runSel"), ("/quad", "/runs"), ("/compare", "/runs")):
        code, body, _ = _http(port, page)
        assert code == 200 and marker in body.decode("utf-8"), page
    # unknown run / non-run entries / traversal -> 404, never a file outside the run
    for bad in ("/r/nope/meta", "/r/notarun/meta", "/r/loose.txt/meta", "/r/../meta",
                "/r/..%2F/meta", "/r/runA%2F..%2FrunA/meta", "/r/runA/../../runA/meta",
                "/r/runA%5C..%5CrunA/meta", "/r//meta", "/r/runA", "/r/runA/state/x"):
        assert _http(port, bad)[0] == 404, bad
    # truncated (mid-write) packet -> 503 after retries; the handler survives
    (root / "runA" / "state.bin").write_bytes(b"\x10\x00\x00\x00{\"seq\": 9")
    assert _http(port, "/r/runA/state")[0] == 503
    assert _http(port, "/runs")[0] == 200
    # restart flag round trip through FileHub
    code, body, _ = _http(port, "/r/runA/restart", method="POST")
    assert code == 200 and json.loads(body) == {"ok": True}
    assert (root / "runA" / "restart.flag").exists()
    assert hub.poll_restart() and not (root / "runA" / "restart.flag").exists()
    assert _http(port, "/r/nope/restart", method="POST")[0] == 404
    assert _http(port, "/r/runA/meta", method="POST")[0] == 404


def test_pages_reference_multi_run_endpoints():
    viewer = ROOT / "physmorph" / "viewer"
    live = (viewer / "live.html").read_text(encoding="utf-8")
    for needle in ('id="runSel"', 'id="scrub"', "'/runs'", "base()", "/commit/",
                   'id="age"', 'id="liveBtn"', 'id="playBtn"'):
        assert needle in live, needle
    for page in ("quad.html", "compare.html"):
        html = (viewer / page).read_text(encoding="utf-8")
        assert "/runs" in html and "?run=" in html, page
    assert "ports" in (viewer / "quad.html").read_text(encoding="utf-8")  # legacy mode


def test_viewer_tunnel_probe(served):
    vt = _load_script("viewer_tunnel")
    _, port, _, _ = served
    assert vt.probe(port) is True
    assert vt.main(["--probe", "--port", str(port)]) == 0
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    closed = s.getsockname()[1]
    s.close()
    # Windows loopback can take up to the timeout to report a closed port: keep it short.
    assert vt.probe(closed, timeout=0.3) is False
    assert vt.main(["--probe", "--port", str(closed), "--probe-timeout", "0.3"]) != 0
