"""
Diagnose where F gradient breaks in the upsample → render chain.

Tests each stage independently to find the exact cutoff point.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np

def test_stage(name, fn):
    """Run fn, check if F_t.grad is not None after backward."""
    try:
        result = fn()
        if result is None:
            print(f"  [{name}] SKIP (returned None)")
            return
        has_grad, grad_norm = result
        status = "OK" if has_grad else "BROKEN"
        print(f"  [{name}] {status} — grad_norm={grad_norm:.6e}" if has_grad
              else f"  [{name}] {status} — F.grad is None")
    except Exception as e:
        print(f"  [{name}] ERROR: {e}")


def main():
    N = 100
    device = 'cpu'

    print("=" * 70)
    print("F gradient chain diagnosis")
    print("=" * 70)

    # ── Test 1: Direct F@F^T → loss ──
    def test_FFT():
        F = torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone().requires_grad_(True)
        cov = 0.04 * torch.bmm(F, F.transpose(-2, -1))
        loss = cov.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("Direct F@F^T", test_FFT)

    # ── Test 2: SVD → reconstruct → loss ──
    def test_svd():
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.01 * torch.randn(N, 3, 3)).requires_grad_(True)
        U, S, Vh = torch.linalg.svd(F)
        F_recon = U @ torch.diag_embed(S) @ Vh
        loss = F_recon.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("SVD roundtrip", test_svd)

    # ── Test 3: polar_with_sv_soft_clamp ──
    def test_polar():
        from sampling.geometry.deformation_covariance import polar_with_sv_soft_clamp
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.01 * torch.randn(N, 3, 3)).requires_grad_(True)
        R, H, S_cl = polar_with_sv_soft_clamp(F, s_min=0.3, s_max=None)
        loss = H.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("polar_with_sv_soft_clamp → H.sum()", test_polar)

    # ── Test 4: polar → covariance build ──
    def test_polar_cov():
        from sampling.geometry.deformation_covariance import polar_with_sv_soft_clamp
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.01 * torch.randn(N, 3, 3)).requires_grad_(True)
        R, H, S_cl = polar_with_sv_soft_clamp(F, s_min=0.3, s_max=None)
        sigma0 = 0.2
        Sigma0 = (sigma0 ** 2) * torch.eye(3).unsqueeze(0).expand(N, -1, -1)
        cov = torch.bmm(torch.bmm(H, Sigma0), H)
        loss = cov.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("polar → H·Σ₀·H → cov.sum()", test_polar_cov)

    # ── Test 5: KNN indexing F[idx] ──
    def test_knn_index():
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.01 * torch.randn(N, 3, 3)).requires_grad_(True)
        # Simulate KNN: each point picks k=4 neighbors
        idx = torch.randint(0, N, (N, 4))
        w = torch.softmax(torch.randn(N, 4), dim=1)
        F_neighbors = F[idx]  # (N, 4, 3, 3)
        F_interp = torch.einsum('mk,mkrc->mrc', w, F_neighbors)
        loss = F_interp.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("F[idx] gather → einsum → sum()", test_knn_index)

    # ── Test 6: Full build_deformation_covariance ──
    def test_full_build():
        from sampling.geometry.deformation_covariance import build_deformation_covariance
        from sampling.analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE

        knn = HybridFAISSKNN(use_faiss=FAISS_AVAILABLE, tau=0.15)
        x = torch.randn(N, 3)
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.05 * torch.randn(N, 3, 3)).requires_grad_(True)

        cov_cfg = {"sigma0": 0.2, "k_F": 8, "use_polar_decomposition": True,
                   "use_F_smoothing": False, "sv_min": 0.3}
        cov, F_interp, idx = build_deformation_covariance(
            points=x, x_low=x, F_low=F, knn=knn, cfg=cov_cfg
        )
        loss = cov.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("Full build_deformation_covariance", test_full_build)

    # ── Test 7: Full upsample (direct mode) ──
    def test_full_upsample():
        from sampling import default_cfg, upsample

        x = torch.randn(N, 3)
        F = (torch.eye(3).unsqueeze(0).expand(N, -1, -1) + 0.05 * torch.randn(N, 3, 3)).requires_grad_(True)

        cfg = default_cfg()
        cfg['upsample'] = cfg.get('upsample', {})
        cfg['upsample']['covariance'] = {
            'sigma0': 0.2, 'enable_subdivision': False,
            'use_F_smoothing': False, 'use_polar_decomposition': True,
            'sv_min': 0.3,
        }

        result = upsample(x, F, cfg=cfg, return_torch=True, current_episode=0)
        cov = result['cov']
        loss = cov.sum()
        loss.backward()
        return F.grad is not None, F.grad.norm().item() if F.grad is not None else 0
    test_stage("Full upsample (direct mode)", test_full_upsample)

    print("\nDone.")


if __name__ == '__main__':
    main()
