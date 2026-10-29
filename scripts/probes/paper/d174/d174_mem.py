"""d174_mem.py OUT_JSON [pipeline_run args] -- D174 memory track: where a run's GPU memory goes.
Runs the pipeline (pipeline_run's prepare, config and run_main; give --animations 2 --settle_windows 1) with hooks that record,
at every phase of a window (built, tape built, after the taped forward, after the gradient, after the line search, before
the commit, after it), torch's allocated / peak-since-the-last-hook / reserved bytes, Warp's mempool current / high-water,
CuPy's pool and the device's used memory. Before the first two commits it walks every object of the run (the start state,
the target pack, the eval trajectory, the tape, the optimiser's leaves and moments, the objective) and sums the bytes of
every distinct device buffer (torch tensors, Warp arrays and their grads, CuPy arrays) by owner and attribute, each buffer
counted once (first owner wins), split into the part whose size is a multiple of N (per particle) and the rest."""
import json
import os
import sys
import time
from collections import defaultdict

import physmorph  # noqa: F401
import torch
import warp as wp

OUT = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import pipeline_run as P  # noqa: E402
import physmorph.pipeline.run.runner as RR  # noqa: E402
import physmorph.pipeline.window.solve as WS  # noqa: E402
import physmorph.pipeline.window.setup as WSET  # noqa: E402
import physmorph.pipeline.window.rollout as WR  # noqa: E402

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None
REC = {"phases": [], "walks": [], "N": None}
DEV = "cuda:0"


def snap(label):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    r = dict(label=label, t=time.time(), torch_alloc=torch.cuda.memory_allocated(), torch_peak=torch.cuda.max_memory_allocated(),
             torch_reserved=torch.cuda.memory_reserved(), warp_cur=wp.get_mempool_used_mem_current(DEV),
             warp_high=wp.get_mempool_used_mem_high(DEV), device_used=total - free,
             cupy_used=cp.get_default_memory_pool().used_bytes() if cp else 0,
             cupy_total=cp.get_default_memory_pool().total_bytes() if cp else 0)
    torch.cuda.reset_peak_memory_stats()
    REC["phases"].append(r)
    print(f"[mem] {label:<34s} torch {r['torch_alloc'] / 2**30:6.2f} (peak {r['torch_peak'] / 2**30:6.2f}, reserved "
          f"{r['torch_reserved'] / 2**30:6.2f}) warp {r['warp_cur'] / 2**30:6.2f} (high {r['warp_high'] / 2**30:6.2f}) cupy "
          f"{r['cupy_total'] / 2**30:5.2f} device {r['device_used'] / 2**30:6.2f} GiB", flush=True)


def _buffers(obj, path, out, seen_ids, depth=0):
    """(ptr, nbytes, numel, path, is_grad) of every device buffer reachable from obj."""
    if depth > 7 or id(obj) in seen_ids:
        return
    seen_ids.add(id(obj))
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            st = obj.untyped_storage()
            out.append((st.data_ptr(), st.nbytes(), obj.numel(), path, False))
            if obj.grad is not None:
                _buffers(obj.grad, path + ".grad", out, seen_ids, depth + 1)
        return
    if isinstance(obj, wp.array):
        if str(obj.device).startswith("cuda") and obj.ptr:
            out.append((int(obj.ptr), int(obj.capacity or 0), int(obj.size), path, path.endswith(".grad")))
        if getattr(obj, "grad", None) is not None:
            _buffers(obj.grad, path + ".grad", out, seen_ids, depth + 1)
        return
    if cp is not None and isinstance(obj, cp.ndarray):
        out.append((int(obj.data.ptr), int(obj.nbytes), int(obj.size), path, False))
        return
    if isinstance(obj, (str, bytes, int, float, bool, type(None))) or callable(obj) and not hasattr(obj, "__dict__"):
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:20000]:
            _buffers(v, f"{path}[{str(k)[:24]}]" if not isinstance(k, (torch.Tensor, wp.array)) else f"{path}[key]", out, seen_ids, depth + 1)
            if isinstance(k, (torch.Tensor, wp.array)):
                _buffers(k, f"{path}[key]", out, seen_ids, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(list(obj)[:20000]):
            _buffers(v, f"{path}[]", out, seen_ids, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if d is None or type(obj).__module__.startswith(("torch", "warp", "numpy", "builtins")) and not isinstance(obj, wp.Tape):
        return
    for k, v in list(d.items()):
        _buffers(v, f"{path}.{k}", out, seen_ids, depth + 1)


class Cover:
    """The union of the byte ranges seen so far: add() returns the bytes of [p, p + n) not yet covered (views of one
    allocation, e.g. Warp arrays on slices of a torch tensor, are counted once)."""

    def __init__(self):
        self.iv = []                                      # sorted, disjoint [start, end)

    def add(self, p, n):
        import bisect
        a, b = p, p + n
        i = bisect.bisect_left(self.iv, (a, a))
        if i > 0 and self.iv[i - 1][1] > a:
            i -= 1
        new, j, lo, hi = 0, i, a, b
        cur = a
        while j < len(self.iv) and self.iv[j][0] < b:
            s, e = self.iv[j]
            if s > cur:
                new += s - cur
            cur = max(cur, e)
            lo, hi = min(lo, s), max(hi, e)
            j += 1
        if cur < b:
            new += b - cur
        self.iv[i:j] = [(lo, hi)]
        return new


def walk(label, roots, N):
    cover, by = Cover(), defaultdict(lambda: [0, 0])
    owners = defaultdict(lambda: [0, 0])
    for name, obj in roots:
        out = []
        _buffers(obj, name, out, set())
        for ptr, nb0, numel, path, g in out:
            nb = cover.add(ptr, nb0) if nb0 > 0 else 0
            if nb <= 0:
                continue
            per = numel >= N and numel % N == 0
            key = ".".join(path.split(".")[:3]).replace("[]", "")
            by[(key, per)][0] += nb
            by[(key, per)][1] += 1
            owners[(name, per)][0] += nb
            owners[(name, per)][1] += 1
    w = dict(label=label, N=N, owners={f"{k[0]}|{'perN' if k[1] else 'rest'}": v for k, v in owners.items()},
             items={f"{k[0]}|{'perN' if k[1] else 'rest'}": v for k, v in by.items()})
    REC["walks"].append(w)
    tot = sum(v[0] for v in owners.values())
    print(f"[walk] {label}: {tot / 2**30:.2f} GiB in {sum(v[1] for v in owners.values())} buffers", flush=True)
    for k, v in sorted(owners.items(), key=lambda kv: -kv[1][0]):
        print(f"[walk]   {k[0]:<16s} {'perN' if k[1] else 'rest'} {v[0] / 2**30:7.3f} GiB ({v[1]} buffers, {v[0] / N:8.1f} B/particle)", flush=True)
    for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"[walk]     {k[0]:<40s} {'perN' if k[1] else 'rest'} {v[0] / 2**30:7.3f} GiB ({v[1]})", flush=True)


ST = {"win": 0, "tgt": None, "start": None, "run": None}
_orig_init, _orig_grad, _orig_ls, _orig_commit = (WS.WindowOptimizer.__init__, WS.WindowOptimizer.gradient,
                                                 WS.WindowOptimizer.line_search, WS.WindowOptimizer.commit)
_orig_adj, _orig_graph = WSET.Window.adjoint, WS.graph_terms
_orig_bt = RR.build_target


def bt(*a, **k):
    snap("before build_target")
    out = _orig_bt(*a, **k)
    ST["tgt"] = out
    snap("after build_target")
    return out


def init(self, start, prm, cfg, tgt, *a, **k):
    snap(f"w{ST['win']} before window")
    _orig_init(self, start, prm, cfg, tgt, *a, **k)
    ST["start"] = start
    REC["N"] = self.win.N
    snap(f"w{ST['win']} window + objective built")


def adjoint(self):
    first = self._adj is None
    out = _orig_adj(self)
    if first:
        snap(f"w{ST['win']} tape built")
    return out


def graph(*a, **k):
    out = _orig_graph(*a, **k)
    snap(f"w{ST['win']} taped forward")
    return out


def gradient(self, e, it):
    out = _orig_grad(self, e, it)
    snap(f"w{ST['win']} it{it} gradient (3 sweeps)")
    return out


def line_search(self, *a, **k):
    out = _orig_ls(self, *a, **k)
    snap(f"w{ST['win']} it{self._it} line search")
    return out


def commit(self, *a, **k):
    if ST["win"] < 2:
        roots = [("start", ST["start"]), ("target", self.tgt), ("eval_traj", self.win.tr), ("tape", self.win._adj),
                 ("window", self.win), ("optimizer", [self.leaves, self.mom, self.vel, self.u_age, self.rigid]),
                 ("objective", self.obj)]
        walk(f"w{ST['win']} before commit", roots, self.win.N)
    snap(f"w{ST['win']} before commit")
    out = _orig_commit(self, *a, **k)
    snap(f"w{ST['win']} after commit")
    ST["win"] += 1
    return out


RR.build_target = bt
WS.WindowOptimizer.__init__ = init
WS.WindowOptimizer.gradient = gradient
WS.WindowOptimizer.line_search = line_search
WS.WindowOptimizer.commit = commit
WSET.Window.adjoint = adjoint
WS.graph_terms = graph

args = P.parse_args()
torch.cuda.init()
snap("start")
cfg0, prep = P.prepared(args, args.seed)
src, tgt, prm = prep.src, prep.tgt, prep.prm
cfg = P.run_config(args, cfg0, prep)
snap("prepared")
try:
    P.run_main(args, prep, src, tgt, prm, cfg)
finally:
    snap("end")
    json.dump(REC, open(OUT, "w"), indent=1)
    print(f"[mem] written {OUT}", flush=True)
