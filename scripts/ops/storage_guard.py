"""Archive explicit completed failed-run artifacts when this project's data exceeds 100 GB.

Never scans other users' /data trees for deletion. Originals are removed only after
streaming SHA-256 verification of the compressed copy and an unchanged-source check.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/relcfd/chayo/physmorph_v2')
FAILED = [('bp308_bunny', 'bp308_end'), ('bp308d_dragon', 'bp308d_end')]
# Completed P291 failures only. Nested run names keep the same done/log/JSON checks.
FAILED += [(f'c291/{name}', f'c291/{name}.done') for name in (
    'c291_bunny_force60s', 'c291_bunny_terminal60s',
    'c291_bunny_norm8s', 'c291_bunny_normphys8s', 'c291_bunny_force8s',
)]


def sync_directory(path):
    if os.name == 'posix':  # production guard runs on Linux; Windows CPU tests skip directory fsync
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def digest_stream(stream):
    h = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        h.update(block)
    return h.hexdigest()


def archive_verified(src, root):
    root = root.resolve()
    if src.is_symlink() or not src.resolve().is_relative_to(root / 'output'):
        raise ValueError('refusing a path outside the project output directory')
    before = src.stat()
    archive_dir = root / 'archives' / 'completed_failed_runs'
    if not archive_dir.resolve().is_relative_to(root):
        raise ValueError('archive directory escapes the project')
    archive_dir.mkdir(parents=True, exist_ok=True)
    sync_directory(root)
    sync_directory(archive_dir.parent)
    dest = archive_dir / (src.name + '.gz')
    part = dest.with_suffix('.gz.partial')
    if dest.exists() or part.exists():
        raise FileExistsError(f'archive already exists: {dest}')
    source_hash = hashlib.sha256()
    with src.open('rb') as inp, gzip.open(part, 'wb', compresslevel=1) as out:
        while block := inp.read(8 * 1024 * 1024):
            source_hash.update(block)
            out.write(block)
    with gzip.open(part, 'rb') as inp:
        verified = digest_stream(inp)
    after = src.stat()
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    if verified != source_hash.hexdigest() or not stable:
        raise RuntimeError(f'archive verification failed or source changed: {src}')
    if part.stat().st_size >= before.st_size:
        raise RuntimeError(f'compression did not reduce storage; original retained: {src}')
    os.replace(part, dest)
    with dest.open('r+b') as inp:
        os.fsync(inp.fileno())
    record = dict(source=str(src), archive=str(dest), source_bytes=before.st_size,
                  archive_bytes=dest.stat().st_size, sha256=verified, verified=True)
    manifest = dest.with_suffix('.gz.json')
    with manifest.open('w') as out:
        json.dump(record, out, indent=2); out.flush(); os.fsync(out.fileno())
    sync_directory(archive_dir)
    # Re-check identity immediately before the single-file unlink (no recursive deletion).
    now = src.stat()
    if (now.st_size, now.st_mtime_ns, now.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
        raise RuntimeError('source changed before removal; original retained')
    src.unlink()
    sync_directory(src.parent)
    print(json.dumps(record), flush=True)
    return record


def usage(root):
    return int(subprocess.check_output(['du', '-sb', str(root)], text=True).split()[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch-seconds', type=int, default=0)
    args = p.parse_args()
    if not ROOT.is_dir():
        raise SystemExit('this guard is for the hyde06 project directory only')
    import fcntl
    with (ROOT / 'storage_guard.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.watch_seconds
        while True:
            size = usage(ROOT)
            print(json.dumps(dict(time=time.time(), project_bytes=size, limit=100_000_000_000)), flush=True)
            if size > 100_000_000_000:
                for prefix, done in FAILED:
                    out = ROOT / 'output'
                    src = out / (prefix + '_render_full_dt_iso_nn.npz')
                    if src.exists() and all((out / n).is_file() for n in (done, prefix + '.json', prefix + '.log')):
                        archive_verified(src, ROOT)
                    if usage(ROOT) <= 100_000_000_000:
                        break
                if usage(ROOT) > 100_000_000_000:
                    raise SystemExit('explicit archive candidates exhausted; retain other evidence and review storage')
            if time.monotonic() >= deadline:
                break
            time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
