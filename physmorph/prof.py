"""A diagnostic wall-clock split of a window's work, off unless the run asks for it (`--profile`).

Each timed section synchronises the GPU before and after, so the sections add up to real time and the run is
slower while profiling; with the profile off the context manager does nothing.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch

STATE = {"on": False}
_T: dict = defaultdict(float)
_N: dict = defaultdict(int)


@contextmanager
def timed(name: str):
    if not STATE["on"]:
        yield
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _T[name] += time.perf_counter() - t0
        _N[name] += 1


def count(name: str, k: int = 1) -> None:
    if STATE["on"]:
        _N[name] += k


def take() -> dict:
    """The sections' seconds and call counts since the last take, then reset."""
    out = {"t": dict(_T), "n": dict(_N)}
    _T.clear()
    _N.clear()
    return out
