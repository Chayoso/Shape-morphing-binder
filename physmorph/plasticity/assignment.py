"""Balanced (bijective) assignment transport via kNN auction.

sliced-OT converges to the OT map but slowly (sub-linear), so the morph plateaus
short of the target. A true 1-to-1 assignment (each source particle -> a distinct
target particle) reaches the EXACT target shape (IoU = ceiling) and stays balanced
(no clustering). We solve it with Bertsekas' auction over kNN candidate targets,
which is O(N k iters) and scales to 1e5 particles.

Bertsekas 1988, "The auction algorithm". Candidate pruning keeps it tractable.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def balanced_assignment(x, y, k=24, eps_frac=1e-3, max_rounds=200, seed=0):
    """Return assigned targets `ty` (N,3): ty[i] = y[pi(i)] for a near-bijection pi.

    x: (N,3) source positions, y: (M,3) targets (M resampled to N if needed).
    Maximises total negative squared distance over kNN candidates via auction.
    Falls back to greedy nearest-unused for any source left unassigned.
    """
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    if y.shape[0] != N:
        y = y[rng.integers(0, y.shape[0], N)]
    # kNN candidate targets per source (benefit = -dist^2)
    dist, cand = cKDTree(y).query(x, k=min(k, N))           # (N,k)
    benefit = -(dist ** 2)
    scale = float(np.median(dist) ** 2 + 1e-9)
    eps = eps_frac * scale

    owner = np.full(N, -1, np.int64)                        # target j -> source owner
    price = np.zeros(N, np.float64)                         # target prices
    assign = np.full(N, -1, np.int64)                       # source i -> target j
    unassigned = list(range(N))
    rounds = 0
    while unassigned and rounds < max_rounds:
        rounds += 1
        bids = {}                                           # target -> (bid, source)
        for i in unassigned:
            val = benefit[i] - price[cand[i]]               # net value of each candidate
            order = np.argsort(val)[::-1]
            j1 = cand[i, order[0]]
            v1 = val[order[0]]
            v2 = val[order[1]] if len(order) > 1 else v1 - 1.0
            bid = price[j1] + (v1 - v2) + eps               # raise price by the margin
            if j1 not in bids or bid > bids[j1][0]:
                bids[j1] = (bid, i)
        new_unassigned = []
        taken = set()
        for j, (bid, i) in bids.items():
            price[j] = bid
            prev = owner[j]
            if prev != -1:
                assign[prev] = -1
                new_unassigned.append(prev)
            owner[j] = i
            assign[i] = j
            taken.add(i)
        # sources that bid but lost (their target was outbid same round) stay unassigned
        for i in unassigned:
            if i not in taken and assign[i] == -1:
                new_unassigned.append(i)
        unassigned = list(set(new_unassigned))

    # greedy fallback for stragglers (assign nearest still-free target)
    if unassigned:
        free = list(set(range(N)) - set(owner[owner >= 0].tolist()))
        if free:
            ftree = cKDTree(y[free])
            for i in unassigned:
                kk = ftree.query(x[i], k=1)[1]
                assign[i] = free[int(kk)]
    assign[assign < 0] = np.arange(N)[assign < 0]           # last resort identity
    return y[assign].astype(np.float32)


def assignment_displacement(x, y, k=24, seed=0, step=1.0):
    """Displacement toward the balanced-assigned target (drop-in for sliced-OT)."""
    ty = balanced_assignment(x, y, k=k, seed=seed)
    return (step * (ty - x)).astype(np.float32)
