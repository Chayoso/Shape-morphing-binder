"""Standalone viewer server: serves every run directory under --root (written by
``physmorph.viewer.filehub.FileHub`` / ``LiveServer.to_dir`` / ``--live_dir``) with live
polling and commit replay.  Runs anywhere (hyde06 next to the runs, or locally over a
synced copy), outlives the simulation processes, one port for all runs.  Stdlib only.

hyde06:  setsid nohup $PY scripts/viewer_serve.py --root <live root> --port 8765 \\
             > output/viewer_serve.log 2>&1 < /dev/null &
local:   python scripts/viewer_tunnel.py --open            (docs/viewer.md)

GET  /  /quad  /compare                      pages from physmorph/viewer/
GET  /runs                                   [{name, arm, mtime, age, seq, run, phase,
                                               commits, live}] newest first
GET  /r/<name>/meta|target|state|history     the run files
GET  /r/<name>/commits                       available commit seqs, ascending
GET  /r/<name>/commit/<seq>                  one stored packet
POST /r/<name>/restart                       touches restart.flag (FileHub.poll_restart)
<name> must be one path component that exists directly under root with a meta.json.
Files may be mid-write on a non-atomic FS: reads validate + retry and answer 503, never
crash the handler.  Every response is Cache-Control: no-store.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGE_DIR = Path(__file__).resolve().parents[1] / "physmorph" / "viewer"
PAGES = {"/": "live.html", "/index.html": "live.html", "/live.html": "live.html",
         "/quad": "quad.html", "/quad.html": "quad.html",
         "/compare": "compare.html", "/compare.html": "compare.html"}
LIVE_WINDOW = 120.0          # s since the last state.bin write that still counts as live
READ_TRIES, READ_DELAY = 6, 0.04


def packet_header(blob: bytes) -> dict | None:
    """Header of a complete pack_state packet; None when truncated/inconsistent."""
    if len(blob) < 8:
        return None
    hlen = struct.unpack_from("<I", blob)[0]
    if hlen % 4 or 4 + hlen > len(blob):
        return None
    try:
        hdr = json.loads(blob[4:4 + hlen])
    except ValueError:
        return None
    if not isinstance(hdr, dict):
        return None
    floats = hdr.get("payload_floats")
    if not isinstance(floats, int) or 4 + hlen + 4 * floats != len(blob):
        return None
    return hdr


def _header_only(path: Path) -> dict | None:
    """Header without the payload (for /runs); payload consistency is not checked."""
    try:
        with open(path, "rb") as f:
            raw = f.read(4)
            if len(raw) < 4:
                return None
            hlen = struct.unpack_from("<I", raw)[0]
            if hlen > 1 << 20:
                return None
            hdr = json.loads(f.read(hlen))
        return hdr if isinstance(hdr, dict) else None
    except (OSError, ValueError):
        return None


def read_retry(path: Path, check) -> bytes | None:
    """Bytes of ``path`` once ``check(bytes)`` passes; None if absent or never consistent."""
    for _ in range(READ_TRIES):
        try:
            blob = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:                     # replaced while open (Windows)
            blob = None
        if blob is not None and check(blob):
            return blob
        time.sleep(READ_DELAY)
    return None


def _json_ok(blob: bytes) -> bool:
    try:
        json.loads(blob)
        return True
    except ValueError:
        return False


def _packet_ok(blob: bytes) -> bool:
    return packet_header(blob) is not None


def run_dir(root: Path, name: str) -> Path | None:
    """Run directory for a URL name or None: exactly one path component, directly under
    root (symlinks escaping root rejected via resolve), with meta.json."""
    if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
        return None
    d = root / name
    try:
        if d.resolve().parent != root.resolve() or not (d / "meta.json").is_file():
            return None
    except OSError:
        return None
    return d


def commit_seqs(d: Path) -> list[int]:
    out = []
    try:
        for e in os.scandir(d / "commits"):
            stem, ext = os.path.splitext(e.name)
            if ext == ".bin" and stem.isdigit():
                out.append(int(stem))
    except OSError:
        pass
    return sorted(out)


def list_runs(root: Path) -> list[dict]:
    now = time.time()
    runs = []
    try:
        entries = list(os.scandir(root))
    except OSError:
        return runs
    for e in entries:
        d = Path(e.path)
        if not e.is_dir() or not (d / "meta.json").is_file():
            continue
        try:
            meta = json.loads((d / "meta.json").read_bytes())
        except (OSError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        try:
            mtime = (d / "state.bin").stat().st_mtime
        except OSError:
            try:
                mtime = (d / "meta.json").stat().st_mtime
            except OSError:
                continue
        hdr = _header_only(d / "state.bin") or {}
        runs.append({"name": e.name, "arm": meta.get("arm") or meta.get("pipeline"),
                     "mtime": mtime, "age": max(0.0, now - mtime),
                     "seq": hdr.get("seq"), "run": hdr.get("run"), "phase": hdr.get("phase"),
                     "commits": len(commit_seqs(d)), "live": now - mtime < LIVE_WINDOW})
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def make_handler(root: Path):
    class H(BaseHTTPRequestHandler):
        server_version = "physmorph-viewer/1"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                        # client went away mid-response

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _err(self, code, msg):
            self._json({"error": msg}, code)

        def _file(self, path, check, ctype):
            blob = read_retry(path, check)
            if blob is None:
                self._err(503, f"{path.name} missing or being written")
            else:
                self._send(200, blob, ctype)

        def _run(self, parts):
            return (run_dir(root, unquote(parts[2]))
                    if len(parts) >= 4 and parts[1] == "r" else None)

        def do_GET(self):
            try:
                self._get()
            except Exception as e:          # a bad request must not kill the thread
                self._err(500, f"{type(e).__name__}: {e}")

        def _get(self):
            path = urlsplit(self.path).path
            if path in PAGES:
                self._send(200, (PAGE_DIR / PAGES[path]).read_bytes(),
                           "text/html; charset=utf-8")
                return
            if path == "/runs":
                self._json(list_runs(root))
                return
            parts = path.split("/")         # ['', 'r', name, what, ...]
            d = self._run(parts)
            if d is None:
                self._err(404, "not found")
                return
            what, rest = parts[3], parts[4:]
            octet = "application/octet-stream"
            if what == "meta" and not rest:
                self._file(d / "meta.json", _json_ok, "application/json")
            elif what == "history" and not rest:
                self._file(d / "history.json", _json_ok, "application/json")
            elif what == "target" and not rest:
                self._file(d / "target.bin", _packet_ok, octet)
            elif what == "state" and not rest:
                self._file(d / "state.bin", _packet_ok, octet)
            elif what == "commits" and not rest:
                self._json(commit_seqs(d))
            elif what == "commit" and len(rest) == 1 and rest[0].isdigit():
                self._file(d / "commits" / f"{int(rest[0]):06d}.bin", _packet_ok, octet)
            else:
                self._err(404, "not found")

        def do_POST(self):
            try:
                parts = urlsplit(self.path).path.split("/")
                d = self._run(parts)
                if d is None or len(parts) != 4 or parts[3] != "restart":
                    self._err(404, "not found")
                    return
                (d / "restart.flag").touch()
                self._json({"ok": True})
            except Exception as e:
                self._err(500, f"{type(e).__name__}: {e}")
    return H


class _Server(ThreadingHTTPServer):
    daemon_threads = True


def make_server(root, port: int = 8765, bind: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Bound but not yet serving; port 0 picks an ephemeral port (server_address[1])."""
    return _Server((bind, int(port)), make_handler(Path(root).resolve()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory holding one run dir per run")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="127.0.0.1 = tunnel only (default); 0.0.0.0 exposes the LAN")
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"--root {root} is not a directory", file=sys.stderr)
        return 2
    httpd = make_server(root, args.port, args.bind)
    print(f"[viewer_serve] http://{args.bind}:{httpd.server_address[1]}/  "
          f"root={root.resolve()}  (/quad, /compare)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
