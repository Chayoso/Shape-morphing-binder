# Shape Morphing with DiffMPM

A differentiable Material Point Method (MPM) implementation for 3D shape morphing and deformation. This project enables physics-based optimization to morph one 3D object into another using gradient-based techniques.

## Features

- **Differentiable MPM Simulation**: Physics-based deformation with automatic differentiation support
- **3D Shape Morphing**: Transform one 3D mesh into another (e.g., sphere → bunny, sphere → spot)
- **Runtime Surface Synthesis**: Generate high-resolution surface point clouds from low-resolution particle simulations
- **Multiple Upsampling Strategies**: Geodesic, AABB, and hybrid methods for surface reconstruction
- **Gaussian Splatting Export**: Export results as Gaussian splats for rendering
- **Visualization Tools**: Automatic generation of comparison images and histograms

## Requirements

### Python Dependencies
- Python >= 3.8
- numpy
- pybind11
- pyyaml
- PyMCubes==0.1.4
- pymeshlab==2023.12.post1
- taichi==1.5.0

### Build Requirements
- C++17 compatible compiler
- OpenMP support
- CMake (for building dependencies)

### External Libraries (Included)
- Eigen (linear algebra)
- libigl (geometry processing)
- GLM (mathematics)
- Cereal (serialization)
- happly (PLY file I/O)
- autodiff (automatic differentiation)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd shape-morphing_1.6.0
```

### 2. Install Python Dependencies

using conda:
```bash
conda env create -f environment/environments.yml
conda activate diffmpm_v1.8.0
```

using pip:
```bash 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Go to diff-gaussian-rasterization, simple-knn (for CUDA 12.8v)
```bash 
pip install -e . --no-build-isolation --config-settings editable_mode=compat
```
### 3. Build C++ Bindings

#### Windows

```bash
$env:DIFFMPM_DIAGNOSTICS="1"
python -m pip install -e . --no-build-isolation  
```

Or

```bash
python setup.py build_ext --inplace
```
## Usage

### Basic Usage

Run a shape morphing simulation using a configuration file:

```bash
python run.py -c configs/sphere_to_bunny.yaml
```

### With Visualization
```bash
python run.py -c configs/sphere_to_bunny.yaml --png --png-dpi 160 --png-ptsize 0.6
```

### Available Configurations

- `configs/sphere_to_bunny.yaml`: Morph a sphere into a bunny
- `configs/sphere_to_spot.yaml`: Morph a sphere into Spot (cow)

### Output

For each episode, the following artifacts are generated in the output directory:

1. **comparison.png**: Visual comparison of current state vs. target
2. **axis_hist.png**: Histogram showing point distribution along axes
3. **summary.json**: Detailed statistics and parameters
4. **gaussians.npz**: Gaussian splatting data (means and covariances)
5. **surface.ply**: 3D point cloud of the synthesized surface

Output structure:
```
output/
├── ep000/
│   ├── ep000_comparison.png
│   ├── ep000_axis_hist.png
│   ├── ep000_summary.json
│   ├── ep000_gaussians.npz
│   └── ep000_surface_180000.ply
├── ep001/
│   └── ...
└── ...
```

## Configuration

Configuration files are in YAML format. Key parameters:

### Input/Output
```yaml
input_mesh_path: "assets/isosphere.obj"
target_mesh_path: "assets/bunny.obj"
output_dir: "output/"
```

### Simulation Parameters
```yaml
simulation:
  grid_dx: 1                              # Grid cell size
  points_per_cell_cuberoot: 2             # Particle density
  grid_min_point: [-16.0, -16.0, -16.0]   # Grid bounds
  grid_max_point: [16.0, 16.0, 16.0]
  
  # Material properties (Lamé parameters)
  lam: 38888.89                           # Volume elastic modulus (λ)
  mu: 58333.3                             # Shear elastic modulus (μ)
  density: 75.0
  
  dt: 0.00833333333                       # Time step (1/120 sec)
  drag: 0.5                               # Drag coefficient
  external_force: [0.0, 0.0, 0.0]         # External forces
  smoothing_factor: 0.955
```

### Optimization Settings
```yaml
optimization:
  num_animations: 45                      # Number of episodes
  num_timesteps: 10                       # Time steps per episode
  control_stride: 10                      # Control frame interval
  max_gd_iters: 1                         # Gradient descent iterations
  max_ls_iters: 10                        # Line search iterations
  initial_alpha: 0.01                     # Initial step size
  gd_tol: 0.0001                          # Convergence tolerance
```

### Runtime Surface Synthesis
```yaml
sampling:
  runtime_surface:
    M: 180000                             # Target number of surface points
    thickness: 0.12                       # Shell thickness
    density_gamma: 2.5                    # Density kernel exponent
    surf_jitter_alpha: 0.35               # Surface jittering factor
    
    ed:                                   # Edge-aware diffusion
      enabled: true
      num_nodes: 180
      node_knn: 8
      
    post_equalize:                        # Point cloud equalization
      enabled: true
      iters: 3
      k: 24
      step: 0.45
```

## Project Structure

```
shape-morphing_1.6.0/
├── assets/                    # 3D mesh assets (.obj files)
│   ├── bob.obj
│   ├── bunny.obj
│   ├── isosphere.obj
│   └── spot.obj
├── bind/                      # Python binding code
│   └── bind.cpp
├── configs/                   # Configuration files
│   ├── sphere_to_bunny.yaml
│   └── sphere_to_spot.yaml
├── DiffMPMLib3D/              # Core C++ MPM library
│   ├── CompGraph.cpp/h        # Computation graph
│   ├── ForwardSimulation.cpp/h
│   ├── BackPropagation.cpp/h
│   ├── Elasticity.cpp/h
│   ├── MaterialPoint.cpp/h
│   ├── Grid.cpp/h
│   └── ...
├── include/                   # External libraries
│   ├── Eigen/                 # Linear algebra
│   ├── igl/                   # Geometry processing
│   ├── autodiff/              # Automatic differentiation
│   └── ...
├── sampling/                  # Runtime surface synthesis
│   └── runtime_surface.py
├── environment/               # Dependency specifications
│   ├── requirements.txt
│   └── environments.yml
├── run.py                     # Main runner script
├── setup.py                   # Build script
└── README.md                  # This file
```

## How It Works

### 1. Material Point Method (MPM)
The simulation uses MPM, a hybrid Eulerian-Lagrangian method that combines the strengths of particle-based and grid-based approaches. Material points carry physical properties (mass, velocity, deformation gradient), while a background grid is used for computing forces and updating velocities.

### 2. Differentiable Optimization
The framework implements automatic differentiation through the entire simulation pipeline, enabling gradient-based optimization. The system minimizes the difference between the simulated result and target shape by adjusting control parameters.

### 3. Runtime Surface Synthesis
After optimization, the system generates a high-resolution surface point cloud from the low-resolution particle simulation using:
- **Density estimation** from deformation gradients
- **Edge-aware diffusion** for smooth transitions
- **Adaptive sampling** based on local curvature
- **Hybrid upsampling** combining geodesic and AABB methods

### 4. Multi-Episode Training
The optimization runs multiple episodes, with each episode refining the transformation. Results from each episode are used to initialize the next, progressively improving the morphing quality.

## Performance Notes

- **Compilation**: The project uses aggressive optimization flags (`/O2` on Windows, `-O3` on Linux)
- **Parallelization**: OpenMP is used for multi-threaded computation
- **SIMD**: AVX2 instructions are enabled for vectorized operations
- **Precision**: Default is single precision; use `DIFFMPM_DOUBLE=1` for double precision

## Troubleshooting

### Build Issues

**Problem**: Compiler errors about missing OpenMP
- **Solution**: Install OpenMP support for your compiler (Visual Studio on Windows includes it by default)

**Problem**: Eigen-related compilation errors
- **Solution**: Ensure C++17 support is enabled

### Runtime Issues

**Problem**: "C++ diffmpm_bindings are required" error
- **Solution**: Build the C++ bindings first using `python setup.py build_ext --inplace`

**Problem**: Out of memory errors
- **Solution**: Reduce `M` (target points) in the configuration or decrease grid resolution

## Citation

If you use this work in your research, please cite:

```bibtex
@software{diffmpm_shape_morphing,
  title={Shape Morphing with Differentiable Material Point Method},
  author={Changyong Song},
  version={1.6.0},
  year={2024}
}
```

## License

[Add your license information here]

## Acknowledgments

This project builds upon:
- Material Point Method research
- Differentiable physics simulation
- Various open-source geometry processing libraries (Eigen, libigl, etc.)

## Contact

For questions or issues, please open an issue on the project repository.

