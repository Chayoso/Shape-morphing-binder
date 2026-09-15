"""LOCAL helper (Windows/PowerShell friendly, stdlib only): keeps the ssh tunnel to the
standalone viewer (scripts/viewer_serve.py on hyde06) alive and says when it is usable.

  python scripts/viewer_tunnel.py --open      # tunnel + open http://127.0.0.1:8765/ once up
  python scripts/viewer_tunnel.py --probe     # exit 0 iff /runs answers (for scripts)

Runs ``ssh -N -L <port>:localhost:<remote-port> -J <jump> <host>``, restarts the child
whenever it exits (exponential backoff 2 s .. 60 s), probes http://127.0.0.1:<port>/runs
every --interval s and prints one status line per probe.  Ctrl-C terminates ssh.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser


def probe_runs(port: int, timeout: float = 2.0, host: str = "127.0.0.1"):
    """The /runs list when viewer_serve answers behind the port, else None."""
    try:
        with urllib.request.urlopen(f"http://{host}:{int(port)}/runs", timeout=timeout) as r:
            runs = json.loads(r.read()) if r.status == 200 else None
    except Exception:
        return None
    return runs if isinstance(runs, list) else None


def probe(port: int, timeout: float = 2.0, host: str = "127.0.0.1") -> bool:
    return probe_runs(port, timeout, host) is not None


def ssh_command(args) -> list[str]:
    cmd = [args.ssh, "-N", "-o", "ExitOnForwardFailure=yes",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
           "-L", f"{args.port}:localhost:{args.remote_port}"]
    if args.jump:
        cmd += ["-J", args.jump]
    return cmd + [args.host]


def _status(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(args) -> int:
    child, backoff, opened, last_err = None, 2.0, False, ""
    next_probe = 0.0
    try:
        while True:
            if child is None or child.poll() is not None:
                if child is not None:
                    last_err = f"ssh exited {child.returncode}"
                    _status(f"reconnecting in {backoff:.0f}s ({last_err})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                try:
                    child = subprocess.Popen(ssh_command(args))
                except OSError as e:
                    last_err = f"cannot start ssh: {e}"
                    _status(f"{last_err}; retry in {backoff:.0f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                _status("ssh started: " + " ".join(ssh_command(args)))
                next_probe = time.monotonic() + 1.5
            now = time.monotonic()
            if now >= next_probe:
                runs = probe_runs(args.port, args.probe_timeout)
                if runs is not None:
                    backoff = 2.0
                    live = sum(1 for r in runs if r.get("live"))
                    _status(f"connected http://127.0.0.1:{args.port}/  "
                            f"{len(runs)} runs, {live} live")
                    if args.open and not opened:
                        webbrowser.open(f"http://127.0.0.1:{args.port}/")
                        opened = True
                else:
                    _status("waiting: tunnel up but /runs not answering"
                            + (f" (last error: {last_err})" if last_err else "")
                            + " -- is viewer_serve.py running on the remote port?")
                next_probe = now + args.interval
            time.sleep(0.5)
    except KeyboardInterrupt:
        _status("stopping")
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                child.kill()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765, help="local port")
    ap.add_argument("--remote-port", type=int, default=8765, help="viewer_serve port on host")
    ap.add_argument("--host", default="chayo@hyde06.dabh.io")
    ap.add_argument("--jump", default="chayo@hyde01.dabh.io", help="'' to disable -J")
    ap.add_argument("--ssh", default="ssh", help="ssh executable")
    ap.add_argument("--interval", type=float, default=10.0, help="probe period (s)")
    ap.add_argument("--probe-timeout", type=float, default=2.0)
    ap.add_argument("--open", action="store_true", help="open the browser once /runs answers")
    ap.add_argument("--probe", action="store_true",
                    help="only probe 127.0.0.1:<port>/runs and exit 0/1 (no ssh)")
    args = ap.parse_args(argv)
    if args.probe:
        runs = probe_runs(args.port, args.probe_timeout)
        if runs is None:
            print(f"viewer not reachable on 127.0.0.1:{args.port}")
            return 1
        print(f"ok: {len(runs)} runs, {sum(1 for r in runs if r.get('live'))} live")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
