"""File-backed drop-in for ``server.Hub``: every packet is persisted to a run directory so
the standalone ``scripts/viewer_serve.py`` (any process, any lifetime, one port for all
runs) can serve it live and replay it after the simulation process is gone.

Run-directory protocol (all writes are tmp-file + ``os.replace``, so readers never see a
half-written file on a POSIX/NTFS local FS)::

    meta.json           bytes assigned to ``hub.meta``      (begin_run)
    target.bin          bytes assigned to ``hub.target``    (begin_run)
    state.bin           latest packet                       (every publish)
    history.json        header rows, same 800-row cap as Hub; rewritten at most every
                        ``history_throttle`` s, except a non-iter packet always flushes
    commits/NNNNNN.bin  every packet whose header phase != "iter" (initial/commit),
                        NNNNNN = zero-padded seq; oldest deleted past ``max_commits``
    restart.flag        created by viewer_serve on POST /restart; consumed by poll_restart()

Constraints: disk errors never propagate into the simulation (counted in ``io_errors``);
packets are the unchanged ``pack_state`` bytes; seq is per process, so a second process in
the same directory overwrites commit files by name (use a fresh directory per process).
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
from pathlib import Path

from .server import _json_value

HISTORY_CAP = 800


def packet_header(blob: bytes) -> dict:
    """JSON header of a pack_state packet (``<u32 hlen><json hdr>...``)."""
    hlen = struct.unpack_from("<I", blob)[0]
    return json.loads(blob[4:4 + hlen])


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a same-directory temp file + os.replace.  Windows readers holding the
    target open make replace fail transiently, hence the short retry."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass


class _FlagEvent(threading.Event):
    """threading.Event that also honours a flag file, so a run holding on
    ``hub.restart.wait()`` (no publish() to poll for it) still sees POST /restart."""

    def __init__(self, flag: Path):
        super().__init__()
        self._flag_path = flag          # NOT _flag: threading.Event's own state attribute

    def poll(self) -> bool:
        if os.path.exists(self._flag_path):
            try:
                os.remove(self._flag_path)
            except OSError:
                pass
            self.set()
        return self.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.poll():
                return True
            rem = 0.5 if end is None else min(0.5, end - time.monotonic())
            if rem <= 0:
                return self.is_set()
            if super().wait(rem):
                return True


class FileHub:
    """Same attribute/method surface as ``server.Hub`` (lock, state, meta, target, history,
    restart, publish, snap, hist_json) plus disk persistence.  Not an HTTP server."""

    def __init__(self, run_dir, max_commits: int = 2000, history_throttle: float = 0.5):
        self.run_dir = Path(run_dir)
        self.commits_dir = self.run_dir / "commits"
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.max_commits = int(max_commits)
        self.history_throttle = float(history_throttle)
        self.lock = threading.Lock()
        self.state = b""
        self._meta = b"{}"
        self._target = b""
        self.history: list[dict] = []
        self.restart = _FlagEvent(self.run_dir / "restart.flag")
        self.io_errors = 0
        self._hist_flushed = 0.0
        self._hist_dirty = False
        # Existing files count toward the cap so a resumed directory keeps pruning.
        self._commit_files = sorted((p for p in self.commits_dir.glob("*.bin")
                                     if p.stem.isdigit()), key=lambda p: int(p.stem))

    # -- meta / target: assignment persists ---------------------------------------------
    @property
    def meta(self) -> bytes:
        return self._meta

    @meta.setter
    def meta(self, blob: bytes) -> None:
        self._meta = blob
        self._write(self.run_dir / "meta.json", blob)

    @property
    def target(self) -> bytes:
        return self._target

    @target.setter
    def target(self, blob: bytes) -> None:
        self._target = blob
        self._write(self.run_dir / "target.bin", blob)

    # -- Hub surface ----------------------------------------------------------------------
    def publish(self, blob: bytes, hdr: dict | None = None) -> None:
        with self.lock:
            self.state = blob
            if hdr is not None:
                self.history.append({k: _json_value(v) for k, v in hdr.items()})
                if len(self.history) > HISTORY_CAP:
                    self.history.pop(0)
            self._hist_dirty = True
        try:
            phdr = packet_header(blob)
            phase, seq = str(phdr.get("phase", "commit")), phdr.get("seq")
        except (ValueError, KeyError, struct.error, IndexError, AttributeError):
            phase, seq = "iter", None       # unparseable: keep state.bin only
        self._write(self.run_dir / "state.bin", blob)
        if phase != "iter":
            if isinstance(seq, int):
                self._store_commit(seq, blob)
            self.flush_history(force=True)
        else:
            self.flush_history(force=False)
        self.poll_restart()

    def snap(self) -> bytes:
        with self.lock:
            return self.state

    def hist_json(self) -> bytes:
        with self.lock:
            return json.dumps(self.history, allow_nan=False).encode()

    # -- persistence helpers --------------------------------------------------------------
    def poll_restart(self) -> bool:
        """Consume restart.flag if present (sets the event); returns is_set()."""
        return self.restart.poll()

    def flush_history(self, force: bool = False) -> None:
        now = time.monotonic()
        if not self._hist_dirty:
            return
        if not force and now - self._hist_flushed < self.history_throttle:
            return
        if self._write(self.run_dir / "history.json", self.hist_json()):
            self._hist_flushed = now
            self._hist_dirty = False

    def _store_commit(self, seq: int, blob: bytes) -> None:
        path = self.commits_dir / f"{int(seq):06d}.bin"
        if not self._write(path, blob):
            return
        if path not in self._commit_files:
            self._commit_files.append(path)
            self._commit_files.sort(key=lambda p: int(p.stem))   # numeric, not lexicographic
        while len(self._commit_files) > self.max_commits:
            oldest = self._commit_files.pop(0)
            try:
                os.remove(oldest)
            except OSError:
                pass

    def _write(self, path: Path, data: bytes) -> bool:
        try:
            atomic_write(path, data)
            return True
        except OSError as e:
            self.io_errors += 1
            if self.io_errors == 1:
                print(f"[filehub] write {path.name} failed ({e}); the run continues",
                      flush=True)
            return False
