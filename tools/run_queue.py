#!/usr/bin/env python3
import argparse, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from queue import Queue
from threading import Thread, Lock

# ---------- Utils ----------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def which_python() -> str:
    return os.environ.get("PYTHON", sys.executable or "python")

def to_list_str(x: Optional[List[str]]) -> List[str]:
    return [str(t) for t in (x or [])]

# ---------- Worker ----------
class Job:
    def __init__(self, idx: int, config: Path, tag: Optional[str],
                 sentinel: Optional[Path], extra_args: List[str]):
        self.idx = idx
        self.config = config
        self.tag = tag or config.stem
        self.sentinel = sentinel
        self.extra_args = extra_args

    def __repr__(self):
        return f"Job(idx={self.idx}, tag={self.tag}, cfg={self.config}, sentinel={self.sentinel})"

def run_one(job: Job, args, gpu_id: Optional[str], run_root: Path) -> Dict[str, Any]:
    """
    Returns dict with fields:
    - rc: return code
    - start_ts, end_ts
    - log_dir, log_file
    - cmd
    """
    log_dir = run_root / f"{ts()}__{job.tag}"
    ensure_dir(log_dir)
    log_file = log_dir / "run.log"
    meta_file = log_dir / "meta.json"

    # Completion check: skip if sentinel already exists (Resume mode)
    if args.resume and job.sentinel and job.sentinel.exists():
        with open(log_file, "a", encoding="utf-8") as f:
            print(f"[RESUME] Sentinel exists. Skip: {job.sentinel}", file=f)
        return {"rc": 0, "skipped": True, "log_dir": str(log_dir), "log_file": str(log_file), "cmd": None}

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [which_python(), str(args.runpy), "--config", str(job.config)] + job.extra_args
    start = time.time()

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# CMD: {' '.join(cmd)}\n")
        f.write(f"# CWD: {Path.cwd()}\n")
        f.write(f"# GPU: {gpu_id}\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=args.cwd, env=env)

    end = time.time()

    result = {
        "rc": rc,
        "start_ts": start,
        "end_ts": end,
        "cmd": cmd,
        "log_dir": str(log_dir),
        "log_file": str(log_file),
        "gpu": gpu_id,
        "skipped": False,
    }

    # Check result if sentinel is specified
    if rc == 0 and job.sentinel:
        if not job.sentinel.exists():
            # Warning if successful but output is missing
            result["rc"] = 10  # custom code
            with open(log_file, "a", encoding="utf-8") as f:
                print(f"[WARN] RC=0 but sentinel missing: {job.sentinel}", file=f)

    # Record metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result

# ---------- Scheduler (sequential or limited-parallel) ----------
def schedule(jobs: List[Job], args):
    run_root = (args.log_root or (args.root / "output" / "_runs")).resolve()
    ensure_dir(run_root)

    gpu_list = [g.strip() for g in args.gpus.split(",")] if args.gpus else [""]
    concurrency = min(args.concurrency, len(gpu_list))
    if concurrency < 1:
        concurrency = 1

    print(f"[INFO] Total jobs = {len(jobs)}")
    print(f"[INFO] GPUs = {gpu_list}  (concurrency={concurrency})")
    print(f"[INFO] log_root = {run_root}")

    # Simple worker pool
    q: Queue[Job] = Queue()
    for j in jobs:
        q.put(j)

    results = []
    lock = Lock()

    def worker(gpu_id: Optional[str]):
        while True:
            try:
                job = q.get_nowait()
            except Exception:
                return
            tries = 0
            while True:
                tries += 1
                res = run_one(job, args, gpu_id if gpu_id != "" else None, run_root)
                with lock:
                    results.append(res)
                if res["rc"] == 0 or tries > args.retries:
                    break
                else:
                    print(f"[RETRY] Job {job.tag} failed (rc={res['rc']}), retry {tries}/{args.retries}")
                    time.sleep(args.retry_wait)

    threads = []
    for i in range(concurrency):
        gpu_id = gpu_list[i % len(gpu_list)]
        t = Thread(target=worker, args=(gpu_id,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Summary
    n_ok = sum(1 for r in results if r["rc"] == 0 or r.get("skipped"))
    n_fail = sum(1 for r in results if r["rc"] != 0 and not r.get("skipped"))
    print(f"\n[SUMMARY] ok={n_ok}, fail={n_fail}, total={len(results)}")
    if n_fail > 0:
        print("[SUMMARY] Some jobs failed. See logs above.")

def main():
    parser = argparse.ArgumentParser(description="Queue runner for YAML configs (DiffMPM + GS).")
    parser.add_argument("--queue", type=Path, required=True, help="YAML list file (see example).")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repo root (default: CWD).")
    parser.add_argument("--runpy", type=Path, default=Path("run.py"), help="Entry script (default: run.py).")
    parser.add_argument("--cwd", type=str, default=None, help="Working directory to execute (default: args.root).")
    parser.add_argument("--gpus", type=str, default="0", help="CSV gpu ids. e.g., '0' or '0,1'.")
    parser.add_argument("--concurrency", type=int, default=1, help="Max parallel jobs (<= #gpus).")
    parser.add_argument("--retries", type=int, default=0, help="Retry count per job.")
    parser.add_argument("--retry-wait", type=int, default=5, help="Seconds between retries.")
    parser.add_argument("--log-root", type=Path, default=None, help="Custom log root (default: output/_runs).")
    parser.add_argument("--resume", action="store_true", help="Skip jobs whose sentinel already exists.")
    args = parser.parse_args()

    if args.cwd is None:
        args.cwd = str(args.root.resolve())

    queue_cfg = read_yaml(args.queue)
    jobs_raw = queue_cfg.get("jobs", [])
    if not jobs_raw:
        print(f"[ERROR] No jobs in {args.queue}", file=sys.stderr)
        sys.exit(2)

    jobs: List[Job] = []
    for i, j in enumerate(jobs_raw):
        cfg = Path(j["config"]).resolve()
        tag = j.get("tag")
        sentinel = j.get("sentinel")
        sentinel = Path(sentinel).resolve() if sentinel else None
        extra_args = to_list_str(j.get("extra_args"))
        jobs.append(Job(i, cfg, tag, sentinel, extra_args))

    schedule(jobs, args)

if __name__ == "__main__":
    main()
