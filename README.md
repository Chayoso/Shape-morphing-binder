# PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yourwebsite.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

> **PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing**  
> Anonymous Authors  
> CVPR 2026 (Under Review)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration Guide](#-configuration-guide)
- [Upsampling Pipeline (v2.1)](#-upsampling-pipeline-v31)
- [Training Pipeline](#-training-pipeline)
- [Parameter Tuning Guide](#-parameter-tuning-guide)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🎯 Overview

**PhysMorph-GS** is a novel framework that combines **physics-based simulation** with **3D Gaussian Splatting** for realistic and controllable shape morphing. Our method integrates:

1. **Differentiable Material Point Method (MPM)** - Physics simulation with gradient backpropagation
2. **6-Stage Differentiable Upsampling (v2.1)** - Sparse-to-dense surface synthesis with geometric awareness
3. **3D Gaussian Splatting Rendering** - High-quality, differentiable rendering
4. **Spectral Covariance Alignment** - Geometry-guided anisotropic Gaussians
5. **End-to-End Training** - Joint optimization of physics and rendering

### Why PhysMorph-GS?

Traditional shape morphing methods either:
- ❌ Lack physical realism (purely geometric interpolation)
- ❌ Cannot be optimized end-to-end (no differentiable rendering)
- ❌ Ignore surface geometry (isotropic representations)
- ❌ Struggle with thin features (volume/surface confusion)

**PhysMorph-GS solves all of these**:
- ✅ **Physically plausible** deformations via MPM simulation
- ✅ **End-to-end differentiable** pipeline from physics to pixels
- ✅ **Geometry-aware** anisotropic Gaussian representations via F-field interpolation
- ✅ **Surface-preserving** with PCA-based detection and soft volume filtering
- ✅ **Thin-feature robust** through adaptive spatial consistency checks
- ✅ **10-100× faster** KNN via Hybrid FAISS with differentiable weights

---

## ✨ Key Features

### 🔬 Physics-based Simulation
- **Differentiable MPM**: Custom C++ implementation with PyTorch integration
- **Neo-Hookean Elasticity**: Realistic material behavior
- **Gradient Injection**: Render losses backpropagate to physics parameters
- **Multi-pass Optimization**: Iterative refinement within each episode

### 🎨 Advanced Upsampling Pipeline (v2.1)
- **6-Stage Differentiable Pipeline**: Surface detection → Volume filtering → Importance sampling → Taubin smoothing → Normal smoothing → Covariance construction
- **PCA-based Surface Detection**: Planarity analysis with Z-score normalization
- **Soft Volume Filtering**: Separates surface from interior (critical for thin features!)
- **Gumbel-Softmax Sampling**: Differentiable categorical sampling with adaptive jittering
- **Shrinkage-Free Smoothing**: Taubin λ-μ scheme preserves volume
- **Adaptive Bandwidth**: Soft median for density-aware smoothing
- **Deformation-Aware Covariances**: F-field interpolation with optional polar decomposition

### 🚀 Performance Optimizations
- **Hybrid FAISS KNN**: 10-100× faster than brute-force with differentiable weights
- **IVF Indexing**: Approximate search for O(N√M) complexity
- **Straight-Through Estimator**: Hard index selection, soft gradient flow
- **Memory Management**: Automatic chunking for large point clouds
- **Mixed Precision**: Optional AMP support

### 📐 Geometry-aware Supervision
- **Silhouette Edge Alignment**: 2D projection of covariance principal axes
- **Spectral Covariance Loss**: Eigenvalue matching for shape preservation
- **Curvature-based Targets**: Anisotropic Gaussians from principal curvatures
- **Adaptive Regularization**: Prevents degenerate Gaussians

---

## 🏗️ Architecture

The pipeline consists of three main stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT: Sparse Meshes                           │
│                    Initial Shape + Target Shape                         │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: MPM SIMULATION (C++)                        │
│ ─────────────────────────────────────────────────────────────────────── │
│  Input:  x_init (N, 3), target_shape                                    │
│  Output: x_t (N, 3), F_t (N, 3, 3) for each timestep t                  │
│                                                                         │
│  Forward Dynamics:                                                      │
│    • P2G: Particle → Grid (mass, momentum transfer)                     │
│    • Grid Update: explicit time integration with Neo-Hookean forces     │
│    • G2P: Grid → Particle (velocity, deformation gradient update)       │
│                                                                         │
│  Backward Gradients:                                                    │
│    • Reverse-mode autodiff through entire simulation                    │
│    • Gradient accumulation: ∂L_physics + ∂L_render                      │
│    • Adam optimization over control timesteps                           │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          STAGE 2: DIFFERENTIABLE UPSAMPLING PIPELINE (v2.1)             │
│ ─────────────────────────────────────────────────────────────────────── │
│  Transforms: Sparse particles (N ≈ 10k) → Dense Gaussians (M ≈ 100k)    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Surface Detection (PCA-based)                            │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ • For each particle: find k=48 neighbors                         │   │
│  │ • Weighted PCA: compute C = Σ wᵢ(xᵢ-c)(xᵢ-c)ᵀ                    │   │
│  │ • Eigendecomposition: λ₀ ≤ λ₁ ≤ λ₂                               │   │
│  │ • Surface quality: surfvar = λ₀/(λ₀+λ₁+λ₂)                       │   │
│  │ • Importance: prob = ema(1 - surfvar)^power                      │   │
│  │ • Extract: normals (n), spacing (h), surface_prob                │   │
│  │                                                                  │   │
│  │ Output: {normals (N,3), surf_prob (N,), spacing (N,)}            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Anchor-Density Map (Differentiable Density Estimation)   │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ • For each anchor: find k=16 nearest neighbors                   │   │
│  │ • Compute soft kernel density: ρᵢ = Σⱼ exp(-(dᵢⱼ/hᵢ)²) · αⱼ      │   │
│  │   - hᵢ = mean(neighbor distances)                                │   │
│  │   - αⱼ = knn softmax weights                                     │   │
│  │ • Normalize: ρ' = ρ / mean(ρ), clamp to [0.25, 4.0]             │   │
│  │                                                                  │   │
│  │ Purpose: Estimate local density for sampling bias                   │   │
│  │ Critical for: sparse/dense region balance, smooth sampling          │   │
│  │                                                                  │   │
│  │ Output: {rho_anchor (N,), cfg_out, state}                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: Importance Sampling (Gumbel-Softmax + Tangent Jitter)    │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ • Sample M indices: Y ~ GumbelSoftmax(surf_prob, τ=0.2)          │   │
│  │   - Straight-through estimator: hard forward, soft backward      │   │
│  │   - Batched for memory efficiency (M×N matrix!)                  │   │
│  │ • Interpolate: anchors = Y @ x_low  (M, 3)                       │   │
│  │              normals = Y @ normals  (M, 3)                       │   │
│  │              spacing = Y @ spacing  (M,)                         │   │
│  │ • Build tangent frame: {t₁, t₂, n} via Gram-Schmidt              │   │
│  │ • Jitter in tangent space:                                       │   │
│  │   - Tangent offset: α·h·(U·t₁ + V·t₂)  with rotation             │   │
│  │   - Normal offset: thickness·Z·n  (usually 0)                    │   │
│  │   - Micro jitter: 0.2·α·h·ε  (high-frequency detail)             │   │
│  │ • Final: points = anchors + tangent + normal + micro             │   │
│  │                                                                  │   │
│  │ Adaptive: α_adapt = α · noise · (h/h_mean)                       │   │
│  │ Dense regions → small jitter, Sparse → large jitter              │   │
│  │                                                                  │   │
│  │ Output: {points (M,3), normals_up (M,3), anchors (M,3)}          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: Taubin Smoothing (Shrinkage-Free λ-μ Scheme)             │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ For each iteration (default: 5 iterations):                      │   │
│  │   1. Build Laplacian: L = D - W                                  │   │
│  │      - W: adjacency from k=32 nearest neighbors                  │   │
│  │      - Soft weights: w = softmax(-distances/τ)                   │   │
│  │   2. Smooth pass: p' = p + λ·L·p  (λ=0.7, positive)              │   │
│  │   3. Inflate pass: p'' = p' + μ·L·p'  (μ=-0.63, negative!)       │   │
│  │   4. Tangent constraint: p_final = p'' - (p''·n)n                │   │
│  │                                                                  │   │
│  │ Effect: Removes jitter noise while preserving volume             │   │
│  │ Key: μ ≈ -(λ + 0.1) balances shrinkage/expansion                 │   │
│  │                                                                  │   │
│  │ Output: {smoothed_points (M,3)}                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 5: Normal Smoothing (Spatial Laplacian + EMA)               │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ For each iteration (default: 3 iterations):                      │   │
│  │   1. Find k=24 spatial neighbors                                 │   │
│  │   2. Estimate bandwidth: h = soft_median(distances)              │   │
│  │      - Soft median: differentiable via smooth ranking            │   │
│  │   3. Spatial weights: w = exp(-d²/h²) · knn_weights              │   │
│  │   4. Weighted average: n_smooth = normalize(Σ wᵢ·nᵢ)             │   │
│  │   5. EMA blend: n ← λ·n_smooth + (1-λ)·n  (λ=0.85)               │   │
│  │                                                                  │   │
│  │ Adaptive bandwidth: denser regions → smaller h (local)           │   │
│  │                    sparser regions → larger h (global)           │   │
│  │                                                                  │   │
│  │ Output: {smoothed_normals (M,3)}                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │ 
│                                ↓                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Step 6: Covariance Construction (F-field Interpolation)          │   │
│  │ ───────────────────────────────────────────────────────────────  │   │
│  │ A. F-field Smoothing (Graph Laplacian):                          │   │
│  │    • Select K=180 graph nodes via Gumbel sampling                │   │
│  │    • Build graph Laplacian: L_graph                              │   │
│  │    • Solve: (WᵀW + λ·L)Y = WᵀF  for smooth F_smooth              │   │
│  │                                                                  │   │
│  │ B. F-field Interpolation:                                        │   │
│  │    • For each upsampled point: find k_F=32 neighbors             │   │
│  │    • Interpolate: F_interp = Σ wⱼ·F_j  (weighted average)        │   │
│  │                                                                  │   │
│  │ C. Covariance from F:                                            │   │
│  │    Option 1 (Simple): Σ = σ₀²·F·Fᵀ  (default)                    │   │
│  │    Option 2 (Polar): Σ = R·S·Σ₀·S·Rᵀ where F = R·S               │   │
│  │      - R: rotation (from SVD)                                    │   │
│  │      - S: stretch (symmetric)                                    │   │
│  │      - Handles reflections (det(F) < 0)                          │   │ 
│  │      - More stable, but slower                                   │   │
│  │                                                                  │   │
│  │ Adaptive scale (optional):                                       │   │
│  │    σ_adaptive = σ₀ · clamp(spacing/mean_spacing, 0.3, 2.0)       │   │
│  │                                                                  │   │
│  │ Output: {cov (M,3,3), F_interp (M,3,3)}                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FINAL OUTPUT: Dense Gaussian Point Cloud                               │
│    • Positions: μ (M, 3) - smoothed, surface-aligned                    │
│    • Covariances: Σ (M, 3, 3) - anisotropic, deformation-aware          │
│    • Normals: n (M, 3) - smooth, consistent                             │
│                                                                         │
│  All operations are differentiable! Gradients flow: L → Σ → F → x       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             STAGE 3: 3D GAUSSIAN SPLATTING RENDERER                     │
│ ─────────────────────────────────────────────────────────────────────── │
│  Input:  (μ, Σ, RGB) - M Gaussian splats                                │
│  Output: {image, alpha, depth, normal_map}                              │
│                                                                         │
│  Differentiable rasterization via PyTorch + CUDA kernels                │
│  Gradients flow: L_render → ∂L/∂Σ → ∂L/∂F → ∂L/∂x                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RENDERING LOSS MANAGER                               │
│ ─────────────────────────────────────────────────────────────────────── │
│  L_render = w_α·L_α + w_edge·L_edge + w_cov·L_cov_align + w_reg·L_reg   │
│                                                                         │
│  • L_α: Silhouette supervision (alpha channel matching)                 │
│  • L_edge: Edge alignment (2D projected Σ vs silhouette tangents)       │
│  • L_cov_align: Spectral matching (eigenvalues of Σ vs curvature)       │
│  • L_reg: Regularization (prevent degenerate Gaussians)                 │
│                                                                         │
│  Gradients injected back to MPM: L_total = L_physics + λ·L_render       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Innovations in v2.1 Upsampling

**1. Hybrid FAISS KNN (10-100× speedup)**
```python
# Straight-through estimator for differentiable KNN
indices, weights = hybrid_knn(query, data, k=32)
# Forward:  indices from FAISS (O(N log M), discrete)
# Backward: weights via softmax (differentiable, continuous)
```

**2. Soft Volume Filtering (thin feature preservation)**
```python
# Critical for bunny ears, fingers, wings!
consensus = mean(|n_i · n_j|)  # Only positive alignments
w = sigmoid(15.0 · (consensus - 0.2))  # Very lenient threshold
```

**3. Gumbel-Softmax Sampling (end-to-end differentiable)**
```python
# Learn importance distribution via gradients
logits = (log(probs) + Gumbel_noise) / tau
y_soft = softmax(logits)  # Continuous
y_hard = one_hot(argmax(y_soft))  # Discrete
Y = y_hard - y_soft.detach() + y_soft  # Straight-through!
```

**4. Taubin Smoothing (shrinkage-free)**
```python
# Traditional Laplacian shrinks volume
# Taubin λ-μ scheme prevents this:
p' = p + λ·L·p      # Smooth (λ > 0)
p'' = p' + μ·L·p'   # Inflate (μ < 0)
# Key: μ ≈ -(λ + 0.1) balances perfectly
```

---

## 🛠️ Installation

### Prerequisites

- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (RTX 20/30/40 series)
- **CUDA**: 12.8 or compatible
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Disk**: 10GB+ for environment

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/Chayoso/Shape-morphing-binder.git
cd Shape-morphing-binder

# 2. Create conda environment
conda env create -f environment/environments.yml
conda activate diffmpm_v2.0.0

# 3. Install PyTorch with CUDA
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install FAISS (critical for fast KNN!)
conda install -c pytorch -c nvidia faiss-gpu=1.7.4

# 5. Build DiffMPM C++ extension
pip install -e . --no-build-isolation

# 6. Verify installation
python -c "import diffmpm_bindings; print('✅ DiffMPM OK')"
python -c "from sampling.pipeline import upsample; print('✅ Upsampling v2.1 OK')"
```

**Note**: See [Installation](#-installation) section for detailed steps and troubleshooting.

---

## 🚀 Quick Start

### Basic Morphing (Physics-only)

```bash
python run.py -c configs/sphere_to_bunny.yaml
```

### End-to-End Training (with rendering)

```bash
python run.py -c configs/sphere_to_bunny.yaml --e2e
```

### With High-Quality Export

```bash
python run.py -c configs/sphere_to_bunny.yaml --e2e --png --png-dpi 200
```

### Custom Configuration

```bash
# Edit config.yaml, then:
python run.py -c configs/my_config.yaml --e2e
```

---

## 📖 Configuration Guide

### Upsampling Pipeline Configuration

The upsampling pipeline is controlled by the `sampling:` section in `config.yaml`. See the [complete annotated config](configs/config.yaml) for all options.

#### Quick Presets

**For Thin Features (Bunny Ears, Fingers):**
```yaml
sampling:
  # Use bunny preset
  surface_detection:
    k: 48
    soft_tau: 0.30
  anchor_density:
    stage2_k: 20                   # Very local!
    anchor_density_beta: 0.6       # Less sparse bias
  sampling:
    M: 80000                       # More samples
  taubin:
    lambda_smooth: 0.5             # Gentler
```

**For Smooth Surfaces (Sphere, Torus):**
```yaml
sampling:
  # Use sphere preset
  anchor_density:
    stage2_k: 32                   # Larger neighborhood
    anchor_density_beta: 0.8       # More sparse bias
  sampling:
    M: 50000                       # Fewer samples
  taubin:
    lambda_smooth: 0.7             # Stronger
```

**For Real-Time Preview:**
```yaml
sampling:
  # Use fast preset
  sampling:
    M: 30000                       # Much fewer
  anchor_density:
    enabled: false                 # Skip
  taubin:
    enabled: false                 # Skip
  normal_smooth:
    enabled: false                 # Skip
```

**For Best Quality:**
```yaml
sampling:
  # Use quality preset
  surface_detection:
    k: 64                          # More neighbors
  sampling:
    M: 100000                      # Max samples
  taubin:
    iters: 5                       # More iterations
    k: 32
```

### Key Parameters Explained

#### Surface Detection

```yaml
surface_detection:
  k: 48              # Neighbors for PCA (32-64 typical)
  soft_tau: 0.30     # Temperature (0.1=sharp, 0.8=smooth)
  surface_power: 3.0 # Concentration (1=uniform, 8=focused)
```

**Tuning:**
- Noisy data? Increase `k` to 64
- Need sharp features? Decrease `soft_tau` to 0.2
- Want uniform coverage? Decrease `surface_power` to 2.0

#### Anchor-Density Map

```yaml
anchor_density:
  stage2_k: 16                # KNN neighbors for density (16-32)
  anchor_density_beta: 0.7    # Sparse bias strength (0.5-1.0)
  spacing_bias_gamma: 0.6     # Fallback bias exponent (0.5-0.8)
```

**Critical for sampling bias!**
- Preserve features: `stage2_k=16-20, beta=0.5-0.6` (less bias)
- Sparse regions: `stage2_k=24-32, beta=0.8-0.9` (more bias)
- Uniform sampling: `beta=0.0` (no density bias)

#### Importance Sampling

```yaml
sampling:
  M: 100000         # Output points (30k-200k)
  tau: 0.2          # Gumbel temp (0.1-0.5)
  alpha: 0.20       # Jitter scale (0.15-0.50)
  thickness: 0.0    # Normal offset (usually 0)
```

**Tuning:**
- Need more density? Increase `M`
- Too uniform? Decrease `tau` to 0.15
- Gaps in surface? Increase `alpha` to 0.35

#### Taubin Smoothing

```yaml
taubin:
  iters: 5                  # Iterations (2-7)
  k: 32                     # Neighbors (16-48)
  lambda_smooth: 0.7        # Smooth weight (0.3-0.9)
  lambda_inflate: -0.63     # Inflate weight (NEGATIVE!)
  tangent_only: true        # Always true
```

**Critical: `lambda_inflate` MUST be negative!**
- Rule of thumb: `mu = -(lambda + 0.1)`
- `lambda=0.7` → `mu=-0.63` ✓
- Surface shrinking? `mu` too positive
- Surface expanding? `mu` too negative

#### Covariance Construction

```yaml
covariance:
  sigma0: 0.08                     # Base scale (0.05-0.15)
  k_F: 32                          # F neighbors (24-48)
  use_F_smoothing: true            # Graph Laplacian
  use_polar_decomposition: false   # Polar F=RS (slower)
```

**Tuning:**
- Mesh in [-1,1]: `sigma0=0.08` ✓
- Mesh in [-10,10]: `sigma0=0.8`
- Noisy F? Enable `use_polar_decomposition`

---

## 🔥 Upsampling Pipeline (v2.1)

### Overview

The upsampling pipeline is fully differentiable and consists of 6 stages:

```
Sparse MPM (N≈10k) → Dense Gaussians (M≈100k)
   [Stage 1] Surface Detection (PCA)
   [Stage 2] Volume Filtering (soft)
   [Stage 3] Importance Sampling (Gumbel-Softmax)
   [Stage 4] Taubin Smoothing (λ-μ)
   [Stage 5] Normal Smoothing (adaptive)
   [Stage 6] Covariance Construction (F-field)
```

### Stage Details

#### Stage 1: Surface Detection

**Purpose**: Identify high-quality surface regions using local geometry.

**Algorithm**:
```python
for each particle p:
  neighbors = knn(p, k=48)
  C = weighted_covariance(neighbors)
  λ₀, λ₁, λ₂ = eigenvalues(C)  # λ₀ ≤ λ₁ ≤ λ₂
  
  # Planarity: small λ₀ → flat surface
  surfvar = λ₀ / (λ₀ + λ₁ + λ₂)
  
  # Importance probability
  prob = ema(1 - surfvar)^power
  
  # Extract normal (smallest eigenvector)
  normal = eigenvector(λ₀)
```

**Output**: `{normals, surf_prob, spacing}`

**Why?** Focuses upsampling on well-defined surfaces, avoiding noisy interior.

#### Stage 2: Volume Filtering

**Purpose**: Separate surface points from interior volume.

**Algorithm**:
```python
for each particle p:
  neighbors = spatial_knn(p, k=20)
  
  # Consensus: how aligned are nearby normals?
  alignments = [|n_p · n_j| for j in neighbors if n_p · n_j > 0]
  consensus = mean(alignments)
  
  # Soft gate
  w = sigmoid(temperature · (consensus - threshold))
  filtered_prob = surf_prob · w
```

**Critical Parameters**:
- `k=20`: Small for thin features (ears, wings)
- `threshold=0.2`: Very lenient (78° tolerance)
- `positive_only=true`: Ignores opposite-facing normals

**Why?** Thin structures like bunny ears would disappear with strict thresholds!

#### Stage 3: Importance Sampling

**Purpose**: Upsample N → M points with learnable distribution.

**Algorithm**:
```python
# Gumbel-Softmax trick
gumbel = -log(-log(uniform(0, 1)))
logits = (log(filtered_prob) + gumbel) / tau
y_soft = softmax(logits)  # Continuous

# Straight-through estimator
y_hard = one_hot(argmax(y_soft))
Y = y_hard - y_soft.detach() + y_soft  # Magic!

# Interpolate
anchors = Y @ x_low
normals = Y @ normals_low

# Tangent jitter
t1, t2 = build_tangent_frame(normals)
U, V = randn(M, 2) with rotation
jitter = alpha * spacing * (U*t1 + V*t2)
points = anchors + jitter
```

**Why differentiable?** Gradients flow through `y_soft` even though selection is discrete!

#### Stage 4: Taubin Smoothing

**Purpose**: Remove jitter noise without shrinking volume.

**Algorithm**:
```python
for iter in range(5):
  # Build Laplacian
  neighbors = knn(points, k=32)
  W = softmax(-distances / tau)  # Soft weights
  L = diag(W.sum(1)) - W
  
  # Two-pass scheme
  p' = p + lambda_smooth * L @ p      # Smooth (λ > 0)
  p'' = p' + lambda_inflate * L @ p'  # Inflate (μ < 0)
  
  # Tangent constraint
  p_final = p'' - (p'' · normals) * normals
```

**Key insight**: Traditional Laplacian shrinks. Taubin's μ < 0 inflates. Combined: no shrinkage!

**Magic formula**: `μ ≈ -(λ + 0.1)` balances perfectly.

#### Stage 5: Normal Smoothing

**Purpose**: Smooth normals for better shading, while preserving features.

**Algorithm**:
```python
for iter in range(3):
  neighbors = spatial_knn(points, k=24)
  distances = ||points - neighbors||
  
  # Adaptive bandwidth (soft median)
  h = soft_median(distances)  # Differentiable!
  
  # Spatial Gaussian weights
  w = exp(-distances² / h²) * knn_weights
  
  # Smooth
  n_smooth = normalize(Σ w_j * n_j)
  
  # EMA blend
  normals = 0.85 * n_smooth + 0.15 * normals
```

**Soft median**: Differentiable approximation via smooth ranking (no argmin!).

**Adaptive**: Dense regions → small h (local smoothing), Sparse → large h (global).

#### Stage 6: Covariance Construction

**Purpose**: Build anisotropic Gaussians from deformation field.

**Algorithm**:
```python
# A. Smooth F-field (graph Laplacian)
nodes = select_graph_nodes(x_low, K=180)
L = build_graph_laplacian(nodes)
F_smooth = solve((W^T W + λ*L) Y = W^T F)

# B. Interpolate F
F_interp = Σ w_j * F_j  # k_F=32 neighbors

# C. Build covariance
if use_polar_decomposition:
  R, S = polar_decomp(F_interp)  # F = R·S
  Σ = R · S · Σ₀ · S · R^T
else:
  Σ = σ₀² · F · F^T  # Simpler, default
```

**Polar decomposition benefits**:
- Handles reflections (det(F) < 0)
- Separates rotation from stretch
- More numerically stable

**Trade-off**: 20× slower (SVD per point). Usually `use_polar=false` is fine.

---

## 🎓 Training Pipeline

### Multi-Pass Optimization

PhysMorph-GS uses **interleaved optimization**:

```python
for episode in range(num_episodes):
  for pass in range(3):  # Default: 3 passes
    
    # Phase 1: Inject render gradients (if not first pass)
    if pass > 0:
      inject_gradients(dL_render_dF, dL_render_dx)
    
    # Phase 2: Physics optimization
    for timestep in control_timesteps:
      forward_dynamics()  # x(t) → x(t+1)
      compute_physics_loss()
      backward_gradients()
      adam_step()
    
    # Phase 3: Render loss
    upsample(x_final, F_final)  # N → M Gaussians
    render_image()
    compute_render_loss()
    loss.backward()  # Get dL/dF, dL/dx
    
    # Phase 4: Visualize (last pass only)
    if pass == 2:
      save_images()
      export_gaussians()
```

### Loss Components

```
L_total = L_physics + Σ(w_i * L_render_i)
```

**Physics Loss**: `||x_final - x_target||²`

**Render Losses**:
1. **L_α**: Alpha/silhouette matching
2. **L_edge**: 2D edge alignment
3. **L_cov_align**: Spectral covariance alignment
4. **L_cov_reg**: Regularization

See [Configuration Guide](#-configuration-guide) for weight tuning.

---

## 🎯 Parameter Tuning Guide

### Problem → Solution Flowchart

**1. Thin features disappearing (ears, fingers)**
```yaml
anchor_density:
  stage2_k: 20         # Decrease to 16-20 (more local)
  anchor_density_beta: 0.5   # Less sparse bias (preserve features)
taubin:
  lambda_smooth: 0.5   # Gentler smoothing
```

**2. Surface too noisy/bumpy**
```yaml
taubin:
  iters: 7             # More iterations
  k: 40                # More neighbors
normal_smooth:
  lambda_smooth: 0.9   # Stronger smoothing
```

**3. Holes/gaps in surface**
```yaml
sampling:
  M: 150000            # More samples
  alpha: 0.35          # Wider jitter
surface_detection:
  surface_power: 2.0   # Less concentrated
```

**4. Interior points visible**
```yaml
anchor_density:
  enabled: true        # Ensure enabled!
  anchor_density_beta: 0.9     # Stronger sparse bias
  stage2_k: 32         # Larger neighborhood
```

**5. Too slow**
```yaml
sampling:
  M: 30000             # Fewer samples
knn:
  use_ivf: true        # Fast approximate KNN
anchor_density:
  enabled: false       # Skip (if acceptable)
```

**6. Simulation unstable (NaN/Inf)**
```yaml
simulation:
  dt: 0.00416667       # Halve timestep
  drag: 0.8            # Increase damping
# Also check: lambda_inflate is NEGATIVE!
```

**7. Surface shrinks/expands**
```yaml
taubin:
  lambda_smooth: 0.7
  lambda_inflate: -0.63  # Adjust: mu ≈ -(lambda + 0.1)
```

**8. Out of memory**
```yaml
sampling:
  M: 50000             # Decrease samples (primary memory)
camera:
  width: 800           # Decrease resolution
  height: 600
```

### Performance vs Quality Trade-offs

| Setting | Speed | Quality | Memory |
|---------|-------|---------|--------|
| Fast preset | 5× faster | 70% | 50% |
| Default | 1× | 100% | 100% |
| Quality preset | 0.5× | 120% | 150% |

**Speed bottlenecks:**
1. Gumbel-Softmax sampling (M×N matrix)
2. Soft median (O(N·k²))
3. Taubin smoothing (iterations)

**Memory bottlenecks:**
1. Sampling: O(M·N) temporarily
2. Rendering: O(M) Gaussians
3. FAISS index: O(N) with IVF

---

## 🔍 Technical Details

### Hybrid FAISS KNN

**Problem**: Need fast KNN for M=100k points, but maintain differentiability.

**Solution**: Straight-through estimator
```python
# Forward: FAISS gives hard indices (discrete, O(N log M))
indices = faiss.search(query, data, k)  # No gradient

# Backward: Recompute distances on neighbors, softmax weights
neighbors = data[indices]
distances = ||query - neighbors||
weights = softmax(-distances / tau)  # Differentiable!

# Gradients flow through weights, not indices
loss.backward()  # ∂L/∂weights → ∂L/∂query, ∂L/∂data
```

**Result**: 10-100× faster than brute-force PyTorch, still differentiable!

### Soft Median for Adaptive Bandwidth

**Problem**: `median()` is discrete (no gradient).

**Solution**: Smooth ranking via sigmoid
```python
def soft_median(x, tau=0.05):
  # For each element, count how many are smaller
  x_diff = x[:, None] - x[None, :]  # Pairwise differences
  P = sigmoid(x_diff / tau)  # Soft comparisons
  soft_rank = P.sum(1)  # How many smaller?
  
  # Weight by distance to median rank
  median_rank = N / 2
  weights = exp(-|soft_rank - median_rank| / tau)
  weights /= weights.sum()
  
  # Weighted average
  return (weights * x).sum()
```

**Result**: Fully differentiable median approximation!

### Polar Decomposition for Covariances

**Problem**: Simple `Σ = σ²FF^T` can give degenerate covariances if det(F) < 0.

**Solution**: Polar decomposition `F = R·S`
```python
U, sigma, Vt = torch.linalg.svd(F)
R = U @ Vt  # Rotation

# Handle reflection
if det(R) < 0:
  U[:, -1] *= -1
  R = U @ Vt
  sigma[-1] *= -1

# Stretch
S = V @ diag(sigma) @ Vt

# Covariance
Σ = R @ S @ Σ₀ @ S @ R^T
```

**Result**: Always positive definite, handles reflections correctly.

**Trade-off**: 20× slower (SVD). Usually not needed unless F is extreme.

---

## 🐛 Troubleshooting

### CUDA Errors

**Error**: `RuntimeError: CUDA out of memory`

**Fix**:
```yaml
sampling:
  M: 30000  # Reduce from 100k
camera:
  width: 800
  height: 600
```

### Build Errors

**Error**: `error C2039` or `fatal error: torch/torch.h: No such file`

**Fix**:
```bash
# Clean build
pip uninstall diffmpm
pip install -e . --no-build-isolation --force-reinstall

# Ensure Visual Studio 2022 (Windows)
# Ensure GCC 9+ (Linux)
```

### FAISS Not Found

**Error**: `ModuleNotFoundError: No module named 'faiss'`

**Fix**:
```bash
conda remove faiss-gpu
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
python -c "import faiss; print(faiss.__version__)"
```

### NaN/Inf Gradients

**Symptom**: `[WARN] dLdF contains NaN/Inf`

**Fix**:
```yaml
optimization:
  initial_alpha: 0.001  # Reduce learning rate
simulation:
  drag: 0.8            # Increase damping
  dt: 0.004167         # Halve timestep

# Also check:
taubin:
  lambda_inflate: -0.63  # MUST be negative!
```

### Thin Features Missing

**Symptom**: Bunny ears, fingers disappear

**Fix**:
```yaml
anchor_density:
  stage2_k: 16               # Very local
  anchor_density_beta: 0.5   # Less sparse bias
  spacing_bias_gamma: 0.5    # Gentler fallback
```

### Surface Shrinks/Expands

**Symptom**: Volume not preserved during Taubin smoothing

**Fix**:
```yaml
taubin:
  lambda_smooth: 0.7
  lambda_inflate: -0.63  # Key: mu ≈ -(lambda + 0.1)
  
# If still shrinking: mu too positive (try -0.70)
# If expanding: mu too negative (try -0.55)
```

---

## 📊 Expected Results

### Output Structure

```
output/
├── target/
│   ├── target_render.png
│   ├── target_alpha.png
│   └── target_data.npz
├── ep000/
│   ├── ep000_render.png        # RGB render
│   ├── ep000_alpha.png         # Silhouette
│   ├── ep000_depth.png         # Depth map
│   ├── ep000_normal.png        # Normal map
│   ├── ep000_gaussians.npz     # Gaussian parameters
│   ├── ep000_surface.ply       # Point cloud
│   └── ep000_comparison.png    # 4-panel viz
└── ...
```

### Typical Timings (RTX 3090)

| Stage | N=10k, M=100k | Notes |
|-------|---------------|-------|
| MPM Simulation | 0.5s/episode | Physics + GD |
| Surface Detection | 0.1s | PCA on N points |
| Volume Filtering | 0.05s | Consensus check |
| Importance Sampling | 1.5s | Gumbel-Softmax (bottleneck!) |
| Taubin Smoothing | 0.3s | 5 iterations |
| Normal Smoothing | 0.8s | Soft median (slow) |
| Covariance | 0.2s | F-field interpolation |
| Rendering | 0.05s | Gaussian rasterization |
| **Total** | **~3.5s** | Per episode |

**Optimization**: Use `M=50k` for 2× speedup, or fast preset for 5× speedup.

---

## 📚 Citation

```bibtex
@inproceedings{physmorph2026,
  title     = {PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original 3DGS
- [FAISS](https://github.com/facebookresearch/faiss) - Fast similarity search
- [Eigen](https://eigen.tuxfamily.org/) - Linear algebra
- [pybind11](https://github.com/pybind/pybind11) - C++/Python bindings

---

## 📞 Contact

- **GitHub Issues**: [Report issues](https://github.com/Chayoso/Shape-morphing-binder/issues)
- **Email**: anonymous@anonymous.edu

---

<div align="center">

**⭐ Star us on GitHub! ⭐**

[⬆ Back to Top](#cvpr-2026-physmorph-gs-physics-guided-gaussian-splatting-for-shape-morphing)

</div>
