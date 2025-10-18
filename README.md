# [CVPR 2026] PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

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
- [Training Pipeline](#-training-pipeline)
- [Advanced Usage](#-advanced-usage)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**PhysMorph-GS** is a novel framework that combines **physics-based simulation** with **3D Gaussian Splatting** for realistic and controllable shape morphing. Our method integrates:

1. **Differentiable Material Point Method (MPM)** - Physics simulation with gradient backpropagation
2. **3D Gaussian Splatting Rendering** - High-quality, differentiable rendering
3. **Curvature-aware Supervision** - Geometry-guided covariance alignment
4. **End-to-End Training** - Joint optimization of physics and rendering

### Why PhysMorph-GS?

Traditional shape morphing methods either:
- ❌ Lack physical realism (purely geometric interpolation)
- ❌ Cannot be optimized end-to-end (no differentiable rendering)
- ❌ Ignore surface geometry (isotropic representations)

**PhysMorph-GS solves all of these**:
- ✅ **Physically plausible** deformations via MPM simulation
- ✅ **End-to-end differentiable** pipeline from physics to pixels
- ✅ **Geometry-aware** anisotropic Gaussian representations
- ✅ **Silhouette-guided** optimization for shape accuracy

---

## ✨ Key Features

### 🔬 Physics-based Simulation
- **Differentiable MPM**: Custom C++ implementation with PyTorch integration
- **Neo-Hookean Elasticity**: Realistic material behavior
- **Gradient Injection**: Render losses backpropagate to physics parameters
- **Multi-pass Optimization**: Iterative refinement within each episode

### 🎨 Advanced Rendering
- **3D Gaussian Splatting**: Fast, differentiable rasterization
- **Runtime Surface Synthesis**: Sparse-to-dense upsampling (N → M, where M >> N)
- **Curvature-based Covariance**: Anisotropic Gaussians aligned with surface geometry
- **Multi-modal Outputs**: RGB, alpha, depth, normal maps

### 📐 Geometry-aware Supervision
- **Silhouette Edge Alignment**: 2D projection of covariance principal axes
- **Spectral Covariance Loss**: Eigenvalue matching for shape preservation
- **Adaptive Density Equalization**: Uniform surface coverage
- **Normal Smoothing**: PCA-based surface detection

### ⚡ Performance Optimizations
- **Hybrid FAISS**: 10-100× faster KNN with differentiable weights
- **IVF Indexing**: Inverted file index for large point clouds
- **Mixed Precision**: Optional AMP support
- **OpenMP Parallelization**: Multi-threaded C++ backend

---

## 🏗️ Architecture

The pipeline consists of four main stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Mesh                               │
│                  (Sparse particle cloud)                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MPM Simulation (C++)                         │
│  • Forward dynamics: x(t), F(t) for each timestep               │
│  • Backward gradients: ∂L_physics/∂x, ∂L_physics/∂F             │
│  • Adam optimization over control timesteps                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│          Runtime Surface Synthesis (Differentiable)             │
│  • Surface detection: PCA-based normal estimation               │
│  • Upsampling: x_low (N) → μ_high (M), M >> N                   │
│  • Covariance: Σ = σ₀² F·Fᵀ (from deformation gradients)        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              3D Gaussian Splatting Renderer                     │
│  • Input: (μ, Σ, RGB) - positions, covariances, colors          │
│  • Output: {image, alpha, depth, normal_map}                    │
│  • Differentiable rasterization via PyTorch                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rendering Loss Manager                       │
│  L_render = w_α·L_α + w_edge·L_edge + w_cov·L_cov_align         │
│                                                                 │
│  • L_α: Silhouette supervision (alpha channel)                  │
│  • L_edge: Edge alignment (2D projection of Σ vs silhouette)    │
│  • L_cov_align: Spectral align (Σ_pred vs. Σ_target)            │
│  • Σ_target = curvature-based anisotropic target covariance     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              Gradient Backpropagation Chain                     │
│                                                                 │
│  L_render → ∂L/∂Σ → ∂L/∂F → ∂L/∂x                               │
│             (cov) (def grad) (position)                         │
│                                                                 │
│  These gradients are injected back to MPM simulation:           │
│  L_total = L_physics + λ·L_render                               │
└─────────────────────────────────────────────────────────────────┘
```

### Training Loop (E2E Interleaved Mode)

```python
For each episode:
  Setup: Create computation graph with T timesteps
  
  For each pass (default: 3 passes):
    
    Phase 1: Inject Render Gradients
      • If not first pass: inject ∂L_render/∂F, ∂L_render/∂x from previous pass
      • C++ backend combines: ∂L_total = ∂L_physics + ∂L_render
    
    Phase 2: Physics Optimization
      • For each control timestep t:
        - Forward: simulate dynamics x(t) → x(t+1)
        - Compute: L_physics = ||x_final - x_target||²
        - Backward: compute ∂L_total/∂x(t)
        - Update: Adam step on control forces
    
    Phase 3: Render Loss Computation
      • Upsample final state: (x_low, F_low) → (μ_high, Σ_high)
      • Render: (μ, Σ, RGB) → {image, alpha, depth}
      • Compare with target render: compute L_render
      • Backprop: L_render.backward() to get ∂L/∂F, ∂L/∂x
      • Store gradients for next pass
    
    Phase 4: Visualization (last pass only)
      • Save: rendered images, normal maps, point clouds
      • Export: NPZ (Gaussians), PLY (mesh), PNG (images)
  
  Promote final state to next episode
```

---

## 🛠️ Installation

### Prerequisites

- **OS**: Windows 10/11 (Linux/macOS also supported)
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.5 (RTX 20/30/40 series recommended)
- **CUDA**: 12.8 (or compatible version)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Disk Space**: 10GB+ for environment and dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chayoso/Shape-morphing-binder.git
cd Shape-morphing-binder
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment/environments.yml
conda activate diffmpm_v2.0.0
```

### Step 3: Install PyTorch with CUDA Support

```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verification:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.8.0+cu128
CUDA Available: True
```

### Step 4: Install FAISS for GPU

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

### Step 5: Install Gaussian Splatting Submodules

```bash
# Navigate to each submodule and install
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian-splatting/submodules/fused-ssim
pip install -e gaussian-splatting/submodules/simple-knn
```

**Note**: On Windows, ensure you have **Visual Studio 2022** with C++ build tools installed.

### Step 6: Build DiffMPM C++ Extension

```bash
# Standard build (optimized)
pip install -e . --no-build-isolation

# Or for debugging (with diagnostics)
set DIFFMPM_DIAGNOSTICS=1
pip install -e . --no-build-isolation --force-reinstall
```

**Build Options:**
- `DIFFMPM_DOUBLE=1` - Use double precision (slower but more accurate)
- `DIFFMPM_DETERMINISTIC=1` - Reproducible builds (disables LTO)
- `DIFFMPM_WITH_TORCH=0` - Disable PyTorch integration
- `DIFFMPM_DIAGNOSTICS=1` - Enable debug output

### Step 7: Verify Installation

```bash
python -c "import diffmpm_bindings; print('✅ DiffMPM bindings loaded successfully')"
python -c "from sampling import synthesize_runtime_surface; print('✅ Sampling module OK')"
python -c "from renderer import GSRenderer3DGS; print('✅ Renderer module OK')"
python -c "from loss import E2ELossManager; print('✅ Loss module OK')"
```

---

## 🚀 Quick Start

### Basic Usage (Physics-only)

Run a simple sphere-to-bunny morphing simulation without rendering supervision:

```bash
python run.py -c configs/sphere_to_bunny.yaml
```

### End-to-End Training

Enable E2E mode with rendering losses:

```bash
python run.py -c configs/sphere_to_bunny.yaml --e2e
```

### With PNG Exports

Export high-resolution comparison visualizations:

```bash
python run.py -c configs/sphere_to_bunny.yaml --e2e --png --png-dpi 200 --png-ptsize 0.8
```

### Custom Configuration

Create your own config file (see [Configuration Guide](#-configuration-guide)):

```bash
python run.py -c configs/my_custom_config.yaml --e2e
```

---

## 📖 Configuration Guide

### Configuration File Structure

The YAML configuration file is organized into several sections:

#### 1. Input/Output Paths

```yaml
input_mesh_path: "assets/isosphere.obj"   # Initial shape (OBJ format)
target_mesh_path: "assets/bunny.obj"      # Target shape (OBJ format)
output_dir: "output/"                     # Output directory
```

**Supported Mesh Formats:**
- `.obj` (Wavefront OBJ, most common)
- Meshes should be watertight and manifold
- Typical size: 1000-10000 vertices

#### 2. Simulation Parameters

```yaml
simulation:
  # Grid Configuration
  grid_dx: 1.0                            # Cell size (smaller = higher resolution)
  grid_min_point: [-16.0, -16.0, -16.0]   # Bounding box minimum
  grid_max_point: [16.0, 16.0, 16.0]      # Bounding box maximum
  points_per_cell_cuberoot: 2             # Particles per cell dimension
  
  # Material Properties (Lamé parameters)
  lam: 38888.89                           # λ (first Lamé parameter)
  mu: 58333.3                             # μ (shear modulus)
  density: 75.0                           # Material density
  
  # Time Integration
  dt: 0.00833333333                       # Time step (1/120 sec)
  drag: 0.5                               # Drag coefficient (0-1)
  external_force: [0.0, 0.0, 0.0]         # External force (e.g., gravity)
  smoothing_factor: 0.955                 # Velocity smoothing
```

**Material Presets:**

| Material | λ (lam) | μ (mu) | Density | Description |
|----------|---------|--------|---------|-------------|
| Rubber   | 10000   | 5000   | 50      | Soft, elastic |
| Default  | 38889   | 58333  | 75      | Medium stiffness |
| Stiff    | 100000  | 150000 | 100     | Rigid, minimal deformation |

**Grid Resolution Guidelines:**
- `grid_dx = 1.0`: Fast, low detail (~10K particles)
- `grid_dx = 0.5`: Balanced (~80K particles)
- `grid_dx = 0.25`: High detail (~640K particles, slow)

#### 3. Optimization Settings

```yaml
optimization:
  num_animations: 2                 # Number of training episodes
  num_timesteps: 10                 # Simulation steps per episode
  control_stride: 1                 # Control frame interval (1 = all frames)
  max_gd_iters: 1                   # Gradient descent iterations per timestep
  max_ls_iters: 10                  # Line search iterations
  initial_alpha: 0.01               # Initial learning rate
  gd_tol: 0.0001                    # Gradient tolerance
  
  # E2E Loss Configuration
  loss:
    enabled: true                   # Enable E2E training
    
    # Loss Weights (see Loss Components section)
    w_alpha: 0.5                    # Silhouette loss
    w_depth: 0.2                    # Depth consistency
    w_edge: 0.1                     # Edge alignment
    w_cov_align: 0.3                # Covariance spectral loss
    w_cov_reg: 0.01                 # Covariance regularization
    
    # Regularization
    cov_reg_mode: 'eigenvalue'      # 'eigenvalue', 'trace', or 'frobenius'
    target_cov_scale: 0.02          # Target Gaussian scale
    
    # Schedule
    schedule: 'constant'            # 'constant', 'linear', or 'cosine'
```

**Control Stride Examples:**

| Stride | Timesteps | Control Frames | Optimizations |
|--------|-----------|----------------|---------------|
| 1      | 10        | [0,1,2,...,8]  | 9             |
| 2      | 10        | [0,2,4,6,8]    | 5             |
| 5      | 10        | [0,5]          | 2             |
| 10     | 10        | [0]            | 1             |

#### 4. Runtime Surface Synthesis

```yaml
sampling:
  runtime_surface:
    # Core Settings
    use_hybrid_e2e: true            # Enable hybrid mode
    base_gaussian_scale: 0.02       # Base σ₀
    use_adaptive_scale: true        # Density-adaptive scaling
    use_normal_anisotropy: false    # Anisotropic covariance (experimental)
    
    # FAISS Acceleration
    use_hybrid_faiss: true          # Fast KNN (10-100× speedup)
    use_faiss_ivf: true             # IVF index
    ivf_nlist: 100                  # Number of clusters
    ivf_nprobe: 10                  # Clusters to search
    
    # Surface Detection
    use_surface_detection: true     # Enable surface filtering
    k_surface: 36                   # Neighbors for detection
    thr_percentile: 8.0             # Surface threshold (top 92%)
    
    # Upsampling
    M: 50000                        # Target number of points
    surf_jitter_alpha: 0.6          # Surface jitter amount
    
    # Post-processing
    post_equalize:
      enabled: true                 # Density equalization
      iters: 8                      # Iterations
      k: 32                         # Neighbors
      step: 0.45                    # Step size
      use_mls_projection: true      # MLS surface projection
```

**Upsampling Size Recommendations:**

| Scene Size | M (Points) | Quality | Speed |
|------------|------------|---------|-------|
| Small      | 20000      | Low     | Fast  |
| Medium     | 50000      | Good    | Medium |
| Large      | 100000     | High    | Slow  |
| Very Large | 200000     | Ultra   | Very Slow |

#### 5. Camera and Rendering

```yaml
camera:
  width: 1280                       # Image width
  height: 720                       # Image height
  fx: 475.0                         # Focal length X (pixels)
  fy: 475.0                         # Focal length Y (pixels)
  cx: 640.0                         # Principal point X
  cy: 360.0                         # Principal point Y
  znear: 0.01                       # Near clipping plane
  zfar: 100.0                       # Far clipping plane
  
  lookat:
    eye: [20.0, -25.0, 15.0]        # Camera position
    target: [0.0, 0.0, 0.0]         # Look-at target
    up: [0.0, 0.0, 1.0]             # Up vector (Z-up)

render:
  bg: [1.0, 1.0, 1.0]               # Background color (white)
  particle_color: [0.27, 0.51, 0.71] # Material color (blue)
  
  lighting:
    model: phong                    # Shading model ('phong' or 'flat')
    type: directional               # Light type
    direction: [0.3, -0.5, 0.8]     # Light direction (normalized)
    ambient: 0.18                   # Ambient intensity
    diffuse: 0.85                   # Diffuse intensity
    specular: 0.10                  # Specular intensity
    shininess: 32                   # Specular shininess
```

---

## 🎓 Training Pipeline

### Multi-Pass Optimization

PhysMorph-GS uses a **multi-pass interleaved optimization** strategy:

1. **Pass 1**: Physics optimization only (no render gradients yet)
2. **Pass 2**: Physics optimization + render gradients from Pass 1
3. **Pass 3**: Physics optimization + render gradients from Pass 2 + visualization

This iterative approach allows the physics simulation to gradually incorporate rendering feedback.

### Loss Components

The total render loss is:

```
L_render = w_α·L_α + w_edge·L_edge + w_cov_align·L_cov_align + w_cov_reg·L_cov_reg
```

#### L_α: Silhouette Loss
- **Purpose**: Match alpha channel (silhouette)
- **Formulation**: `L_α = ||α_pred - α_target||₁`
- **Weight Range**: 0.3 - 0.7

#### L_edge: Edge Alignment Loss
- **Purpose**: Align 2D projected covariance with silhouette edges
- **Method**:
  1. Compute Sobel gradients of target alpha
  2. Extract silhouette tangent vectors
  3. Project 3D covariance to 2D screen space
  4. Align principal axes with edge tangents
- **Formulation**: `L_edge = Σ w_i (1 - |v_max · t_hat|)`
- **Weight Range**: 0.05 - 0.2

#### L_cov_align: Covariance Spectral Loss
- **Purpose**: Match eigenvalue spectrum of predicted and target covariances
- **Target Covariance** (Σ★):
  - Computed from surface curvature
  - Anisotropic: narrow along high-curvature directions
  - Formula: `Σ★ = R·diag(s₁², s₂², s₃²)·Rᵀ`
    - `s₁ = σ₀ / (1 + α·|κ₁|)`
    - `s₂ = σ₀ / (1 + α·|κ₂|)`
    - `s₃ = σ₀ · 0.3`
- **Weight Range**: 0.05 - 0.3

#### L_cov_reg: Covariance Regularization
- **Purpose**: Prevent degenerate covariances
- **Modes**:
  - `eigenvalue`: Match eigenvalues to target scale
  - `trace`: Regularize matrix trace
  - `frobenius`: Frobenius norm penalty
- **Weight**: 0.01 (typical)

### Loss Weight Schedule

Three schedule modes are available:

1. **Constant** (default): Fixed weights throughout training
2. **Linear**: Gradually increase silhouette weight
   ```python
   w_α(t) = w_min + (w_max - w_min) · (t / T)
   ```
3. **Cosine**: Smooth transition with cosine annealing
   ```python
   w_α(t) = w_min + (w_max - w_min) · [1 - cos(π(1-t/T))] / 2
   ```

---

## 🔧 Advanced Usage

### Custom Mesh Preparation

Prepare your own meshes for morphing:

```bash
# 1. Convert to OBJ format (if needed)
# Using Blender, MeshLab, or CloudCompare

# 2. Normalize mesh to unit cube
python tools/normalize_mesh.py input.obj output.obj

# 3. Check mesh quality
python tools/surface_check.py output.obj
```

**Mesh Requirements:**
- Watertight (no holes)
- Manifold (no non-manifold edges)
- Single connected component
- Reasonable triangle count (1K-100K)

### Multi-Episode Training

Train multiple episodes with state promotion:

```yaml
optimization:
  num_animations: 5                # 5 episodes
  num_timesteps: 20                # 20 steps per episode
```

The final state of episode N becomes the initial state of episode N+1.

### Batch Processing

Process multiple configurations in parallel:

```bash
# Create a batch script
python scripts/batch_run.py --config-dir configs/batch/ --parallel 4
```

### Export Formats

PhysMorph-GS exports multiple formats per episode:

**Image Outputs:**
- `ep000_render.png` - Final RGB render (8-bit PNG)
- `ep000_alpha.png` - Silhouette/alpha channel
- `ep000_depth.png` - Depth map (16-bit PNG, millimeters)
- `ep000_normal.png` - Normal map (world space)

**Data Outputs:**
- `ep000_gaussians.npz` - Gaussian parameters (μ, Σ, RGB)
  ```python
  data = np.load("ep000_gaussians.npz")
  mu = data['mu']        # (M, 3) positions
  cov = data['cov']      # (M, 3, 3) covariances
  rgb = data['rgb']      # (M, 3) colors
  ```
- `ep000_surface_50000.ply` - Point cloud (PLY format)
- `ep000_summary.json` - Metrics (loss, J_min, J_mean)

**Visualization Outputs:**
- `ep000_comparison.png` - Before/after 4-panel comparison
- `ep000_axis_hist.png` - Covariance eigenvalue distribution

### Integrating with External Renderers

Export Gaussians for use in other 3DGS viewers:

```python
import numpy as np

# Load Gaussians
data = np.load("output/ep000/ep000_gaussians.npz")
mu = data['mu']
cov = data['cov']
rgb = data['rgb']

# Convert to 3DGS standard format
# (scales + rotations instead of full covariance)
from renderer.utils import cov_to_scale_rotation

scales, rotations = cov_to_scale_rotation(cov)

# Export to .ply for SIBR viewer
save_ply_for_sibr("output.ply", mu, scales, rotations, rgb)
```

---

## 🔍 Technical Details

### Differentiable MPM Implementation

Our MPM solver is fully differentiable with respect to:
- Particle positions `x`
- Deformation gradients `F`
- Control forces `f_control`

**Forward Pass:**
```cpp
// Particle-to-Grid (P2G)
for each particle p:
  compute weight w_ip for each grid node i
  transfer mass, momentum to grid nodes

// Grid Update
for each grid node i:
  v_new = (m·v_old + f_ext·dt) / (m + drag·dt)
  apply boundary conditions

// Grid-to-Particle (G2P)
for each particle p:
  interpolate v_new from grid
  update position: x += v·dt
  update deformation gradient: F_new = (I + ∇v·dt)·F_old
```

**Backward Pass:**
```cpp
// Reverse-mode autodiff
// Gradients flow: L → x(T) → F(T) → ... → x(0) → f_control

for t = T downto 0:
  // Backprop through G2P
  ∂L/∂v_grid += ∂L/∂x_particle · ∂x/∂v_grid
  ∂L/∂v_grid += ∂L/∂F_particle · ∂F/∂v_grid
  
  // Backprop through grid update
  ∂L/∂f_ext += ∂L/∂v_grid · dt
  
  // Backprop through P2G
  ∂L/∂x_particle += ∂L/∂v_grid · ∂v_grid/∂x_particle
```

### Runtime Surface Synthesis

The upsampling process transforms sparse physics particles to dense Gaussians:

**Step 1: Surface Detection**
```python
# Use PCA to detect surface particles
for each particle:
  find k_surface nearest neighbors
  compute local PCA
  density_score = λ₃ / (λ₁ + λ₂ + λ₃)  # small λ₃ → surface
  
  if density_score < threshold:
    mark as surface particle
```

**Step 2: Sampling**
```python
# Sample M points on surface particles
for i in range(M):
  select particle p with probability ∝ surface_weight
  perturb position: μ = x_p + jitter·n_p
```

**Step 3: Covariance Estimation**
```python
# Interpolate deformation gradient
F = weighted_average(F_neighbors, weights=gaussian_kernel)

# Compute covariance from F
Σ = σ₀² · F·Fᵀ
```

**Step 4: Post-processing**
```python
# Density equalization (Lloyd relaxation)
for iter in range(8):
  move points toward local centroid
  project to MLS surface
```

### Curvature Estimation

Principal curvatures are estimated via **normal variation analysis**:

```python
def estimate_curvature(mu, normals, k=16):
    # 1. Find k nearest neighbors
    neighbors = knn_search(mu, k)
    
    # 2. Compute normal variation
    Δn = normals[neighbors] - normals[:, None]
    Δx = mu[neighbors] - mu[:, None]
    
    # 3. Estimate curvature: κ ≈ ||Δn|| / ||Δx||
    curvature = ||Δn|| / (||Δx|| + ε)
    
    # 4. Weighted average (closer = more important)
    weights = exp(-||Δx|| / h)
    κ_mean = Σ(weights · curvature) / Σ(weights)
    κ_std = std(curvature)
    
    # 5. Principal curvatures
    κ₁ = κ_mean + 0.5·κ_std
    κ₂ = κ_mean - 0.5·κ_std
    
    return [κ₁, κ₂]
```

### Gradient Flow Analysis

The end-to-end gradient flow is:

```
Pixel Loss → Rasterizer → Gaussians (μ, Σ, RGB)
                              ↓
                        Runtime Surface
                              ↓
                        Deformation Gradient F
                              ↓
                        Particle Positions x
                              ↓
                        MPM Dynamics
                              ↓
                        Control Forces f_control
```

Key insights:
- **Chain Rule**: Each module provides Jacobians for backprop
- **Gradient Clipping**: Clamp to `[-3, 3]` for stability
- **NaN Handling**: Replace NaN/Inf with zeros
- **Accumulation**: Render gradients are **added** to physics gradients

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions:**
- Reduce upsampling size: `M: 50000` → `M: 20000`
- Reduce image resolution: `width: 1280` → `width: 800`
- Enable mixed precision: `use_amp: true`
- Reduce batch size (if using custom batching)

#### 2. C++ Compilation Errors

**Symptoms:**
```
error C2039: 'xxx': is not a member of 'std'
```

**Solutions:**
- Ensure Visual Studio 2022 is installed (Windows)
- Update CMake: `pip install --upgrade cmake`
- Clean build: `pip uninstall diffmpm && pip install -e . --force-reinstall`

#### 3. FAISS Import Error

**Symptoms:**
```
ImportError: cannot import name 'IndexIVFFlat' from 'faiss'
```

**Solutions:**
```bash
# Reinstall FAISS
conda remove faiss-gpu
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

#### 4. NaN/Inf in Gradients

**Symptoms:**
```
[WARN] dLdF contains NaN/Inf values
```

**Solutions:**
- Reduce learning rate: `initial_alpha: 0.01` → `0.001`
- Increase drag: `drag: 0.5` → `0.8`
- Enable gradient clipping (automatic in loss.py)
- Check mesh quality with `tools/surface_check.py`

#### 5. Rendering Artifacts

**Symptoms:**
- Black spots in render
- Stretched Gaussians

**Solutions:**
- Increase equalization iterations: `iters: 8` → `iters: 12`
- Enable MLS projection: `use_mls_projection: true`
- Reduce `base_gaussian_scale`: `0.02` → `0.015`

### Performance Optimization Tips

#### Speed Up Training
1. **Use IVF indexing**: `use_faiss_ivf: true`
2. **Reduce timesteps**: `num_timesteps: 10` → `5`
3. **Increase control stride**: `control_stride: 1` → `2`
4. **Disable PNG export**: `--png` flag removed

#### Improve Quality
1. **Increase upsampling**: `M: 50000` → `100000`
2. **More equalization**: `iters: 8` → `16`
3. **Finer grid**: `grid_dx: 1.0` → `0.5`
4. **More passes**: Modify `num_passes = 3` → `5` in run.py

#### Memory Usage
- **CPU RAM**: Scales with grid resolution (~100MB per 1M cells)
- **GPU VRAM**: Scales with M (~50MB per 10K Gaussians)

**Estimated VRAM Usage:**

| M (Points) | Resolution | VRAM |
|------------|------------|------|
| 20K        | 800×600    | 2GB  |
| 50K        | 1280×720   | 4GB  |
| 100K       | 1920×1080  | 8GB  |
| 200K       | 2560×1440  | 16GB |

---

## 📊 Expected Output Structure

After running `python run.py -c config.yaml --e2e --png`, you'll get:

```
output/
├── target/
│   ├── target_image.png         # Target shape RGB render
│   ├── target_alpha.png         # Target silhouette
│   ├── target_depth.png         # Target depth (16-bit)
│   ├── target_normal.png        # Target normal map
│   └── target_render.npz        # All target data (NumPy)
├── ep000/
│   ├── ep000_render.png         # Episode 0 final render
│   ├── ep000_alpha.png          # Episode 0 silhouette
│   ├── ep000_depth.png          # Episode 0 depth
│   ├── ep000_normal.png         # Episode 0 normal map
│   ├── ep000_gaussians.npz      # Gaussian parameters (μ, Σ, RGB)
│   ├── ep000_surface_50000.ply  # Point cloud (PLY format)
│   ├── ep000_comparison.png     # 4-panel visualization
│   ├── ep000_axis_hist.png      # Eigenvalue distribution
│   └── ep000_summary.json       # Metrics (loss, J_min, J_mean)
├── ep001/
│   └── ...
└── ...
```

---

## 📚 Citation

If you find this work useful, please cite:

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

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Libraries

This project includes code from:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original 3DGS implementation
- [Eigen](https://eigen.tuxfamily.org/) - Linear algebra library
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- [pybind11](https://github.com/pybind/pybind11) - C++/Python bindings

---

## 🙏 Acknowledgments

We thank:
- The **3D Gaussian Splatting** team for the original rasterization implementation
- The **Taichi** community for inspiration on differentiable physics
- NVIDIA for CUDA and GPU resources
- Anonymous reviewers for valuable feedback

---

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Open an issue](https://github.com/anonymous/physmorph-gs/issues)
- **Email**: anonymous@anonymous.edu
- **Project Page**: https://yourwebsite.com

---

## 🔄 Changelog

### Version 2.0.0 (Current)
- ✨ **NEW**: End-to-end training with rendering supervision
- ✨ **NEW**: Curvature-based target covariance (Σ★)
- ✨ **NEW**: Silhouette edge alignment loss
- ⚡ **IMPROVED**: 10-100× faster KNN with Hybrid FAISS
- ⚡ **IMPROVED**: MLS surface projection for smoother results
- 🐛 **FIXED**: Gradient NaN/Inf issues in backpropagation
- 🐛 **FIXED**: Memory leaks in C++ backend

### Version 1.0.0 (Legacy)
- Initial physics-only implementation
- Basic MPM simulation
- Non-differentiable rendering

---

<div align="center">

**⭐ Star us on GitHub if you find this useful! ⭐**

[⬆ Back to Top](#cvpr-2026-physmorph-gs-physics-guided-gaussian-splatting-for-shape-morphing)

</div>

