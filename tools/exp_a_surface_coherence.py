"""
Exp A: Surface-Aware Gradient Coherence Study

Tests whether render gradient coherence improves when restricted to
surface-aware / visually relevant particle subsets.

Runs fully offline from saved dLdx.npy + x_particles.npy (no physics needed).

Usage:
    python tools/exp_a_surface_coherence.py \
        --dLdx output/sphere_to_bunny/ep010/dLdx.npy \
        --x    output/sphere_to_bunny/ep010/x_particles.npy \
        --out  output/exp_a_results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


# ── Coherence ─────────────────────────────────────────────────────────────

def compute_coherence(g, x_pos, k=16, subsample=15000, seed=42):
    """Avg cosine similarity between neighboring particle gradients (subsampled)."""
    np.random.seed(seed)
    N = len(x_pos)
    if N > subsample:
        idx_sub = np.random.choice(N, subsample, replace=False)
        x_s = x_pos[idx_sub]
        g_s = g[idx_sub]
    else:
        x_s, g_s = x_pos, g

    norms_full = np.linalg.norm(g, axis=1, keepdims=True)
    g_hat_full = g / (norms_full + 1e-8)

    norms_s = np.linalg.norm(g_s, axis=1, keepdims=True)
    g_hat_s = g_s / (norms_s + 1e-8)

    tree = cKDTree(x_pos)
    _, idx_nn = tree.query(x_s, k=min(k + 1, N))

    cos_sims = []
    for i in range(len(x_s)):
        neighbors = idx_nn[i, 1:]
        g_n = g_hat_full[neighbors]
        cos = (g_hat_s[i] * g_n).sum(axis=1)
        cos_sims.append(float(cos.mean()))
    return float(np.mean(cos_sims))


def grad_stats(g):
    norms = np.linalg.norm(g, axis=1)
    return {
        'N': len(g),
        'norm_total': float(np.linalg.norm(g)),
        'per_mean': float(norms.mean()),
        'per_median': float(np.median(norms)),
        'per_p95': float(np.percentile(norms, 95)),
        'per_max': float(norms.max()),
        'spike_ratio': float(norms.max() / (norms.mean() + 1e-12)),
        'nonzero_frac': float((norms > 1e-10).mean()),
    }


# ── Smoothing ─────────────────────────────────────────────────────────────

def smooth_displacement(g, x_pos, k):
    tree = cKDTree(x_pos)
    _, idx = tree.query(x_pos, k=min(k + 1, len(x_pos)))
    return g[idx].mean(axis=1).astype(np.float32)


def clip_outliers(g, pct=95):
    norms = np.linalg.norm(g, axis=1)
    thresh = np.percentile(norms, pct)
    scale = np.minimum(1.0, thresh / (norms + 1e-8))
    return g * scale[:, None]


def analyze_subset(g_sub, x_sub, label):
    """Compute coherence + stats for multiple smoothing variants."""
    print(f"\n  [{label}]  N_subset={len(g_sub):,} ({100*len(g_sub)/len(g_sub):.0f}%)")
    s = grad_stats(g_sub)
    print(f"    per_mean={s['per_mean']:.5f}, max={s['per_max']:.3f}, spike={s['spike_ratio']:.0f}x")

    variants = {'raw': g_sub}
    variants['clip_p95'] = clip_outliers(g_sub, 95)
    variants['knn_k8']   = smooth_displacement(g_sub, x_sub, k=8)
    variants['knn_k32']  = smooth_displacement(g_sub, x_sub, k=32)
    variants['clip_p95+knn_k8']  = smooth_displacement(clip_outliers(g_sub, 95), x_sub, k=8)
    variants['clip_p95+knn_k32'] = smooth_displacement(clip_outliers(g_sub, 95), x_sub, k=32)

    results = {}
    for vname, g_v in variants.items():
        c = compute_coherence(g_v, x_sub)
        st = grad_stats(g_v)
        results[vname] = {'coherence': c, **st}
        print(f"    {vname:30s}: coherence={c:.4f}, spike={st['spike_ratio']:.0f}x, norm={st['norm_total']:.2f}")

    return results


# ── Subset definitions ─────────────────────────────────────────────────────

def subset_nonzero(g):
    """Particles with nonzero render gradient (contributed to alpha loss)."""
    norms = np.linalg.norm(g, axis=1)
    return norms > 1e-10


def subset_top_contrib(g, pct):
    """Top pct% particles by ||dLdx_i||."""
    norms = np.linalg.norm(g, axis=1)
    thresh = np.percentile(norms, 100 - pct)
    return norms >= thresh


def subset_surface_shell(x, k=20, density_pct=30):
    """Low local density = surface/outer layer particles."""
    tree = cKDTree(x)
    dd, _ = tree.query(x, k=min(k + 1, len(x)))
    local_density = 1.0 / (dd[:, 1:].mean(axis=1) + 1e-8)
    thresh = np.percentile(local_density, density_pct)
    return local_density <= thresh


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dLdx', required=True)
    parser.add_argument('--x', required=True)
    parser.add_argument('--out', default='output/exp_a_results.json')
    args = parser.parse_args()

    dLdx = np.load(args.dLdx).astype(np.float32)
    x = np.load(args.x).astype(np.float32)
    N = len(x)

    print(f"Loaded: N={N:,}, dLdx shape={dLdx.shape}")
    print(f"dLdx_norm={np.linalg.norm(dLdx):.2f}, spike_ratio={np.linalg.norm(dLdx, axis=1).max() / np.linalg.norm(dLdx, axis=1).mean():.0f}x")

    all_results = {}

    # ── Full volume baseline ───────────────────────────────────────────────
    print("\n=== Full volume baseline ===")
    all_results['full_volume'] = analyze_subset(dLdx, x, 'full_volume')

    # ── Subset A: nonzero gradient particles ──────────────────────────────
    print("\n=== Subset A: nonzero gradient (visible/contributing) ===")
    mask_nz = subset_nonzero(dLdx)
    print(f"  nonzero: {mask_nz.sum():,} / {N:,} = {100*mask_nz.mean():.1f}%")
    all_results['subset_nonzero'] = analyze_subset(dLdx[mask_nz], x[mask_nz], 'nonzero')

    # ── Subset B: top contribution ────────────────────────────────────────
    for pct in [5, 10, 20]:
        label = f'top_{pct}pct'
        print(f"\n=== Subset B: top {pct}% by ||dLdx_i|| ===")
        mask = subset_top_contrib(dLdx, pct)
        print(f"  {mask.sum():,} particles ({100*mask.mean():.1f}%)")
        all_results[label] = analyze_subset(dLdx[mask], x[mask], label)

    # ── Subset C: surface shell (low local density) ────────────────────────
    print("\n=== Subset C: surface shell (bottom 30% local density) ===")
    print("  Computing local density (k=20)...")
    mask_shell = subset_surface_shell(x, k=20, density_pct=30)
    print(f"  shell: {mask_shell.sum():,} / {N:,} = {100*mask_shell.mean():.1f}%")
    all_results['surface_shell_30pct'] = analyze_subset(dLdx[mask_shell], x[mask_shell], 'surface_shell')

    # ── Subset D: top contrib ∩ surface shell ─────────────────────────────
    print("\n=== Subset D: top_10pct ∩ surface_shell ===")
    mask_combo = subset_top_contrib(dLdx, 10) & mask_shell
    print(f"  intersection: {mask_combo.sum():,} / {N:,} = {100*mask_combo.mean():.1f}%")
    if mask_combo.sum() > 100:
        all_results['top10_x_shell'] = analyze_subset(dLdx[mask_combo], x[mask_combo], 'top10∩shell')
    else:
        print("  Too few particles, skipping")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Best coherence per subset (clip_p95+knn_k32 variant)")
    print(f"{'subset':30s} {'N_sub':>8} {'raw_coh':>10} {'best_coh':>10} {'improvement':>12} {'spike_raw':>10}")
    print("-" * 80)
    for sname, svariants in all_results.items():
        raw_c = svariants['raw']['coherence']
        best_c = max(v['coherence'] for v in svariants.values())
        best_v = max(svariants, key=lambda k: svariants[k]['coherence'])
        n_sub = svariants['raw']['N']
        spike = svariants['raw']['spike_ratio']
        impr = best_c / (svariants['full_volume']['raw']['coherence'] if 'full_volume' in all_results else raw_c + 1e-8) if sname != 'full_volume' else best_c / raw_c
        print(f"  {sname:28s} {n_sub:>8,} {raw_c:>10.4f} {best_c:>10.4f} {impr:>11.1f}x  {spike:>9.0f}x")

    # Fix summary for full_volume
    fv = all_results['full_volume']
    fv_raw_c = fv['raw']['coherence']

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'N_full': N, 'subsets': all_results}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
