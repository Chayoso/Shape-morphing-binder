#!/bin/bash
# Auto-generate remaining experiment configs

cd "$(dirname "$0")"

echo "Creating remaining experiment configs..."

# Function to create config from template
create_config() {
    local num=$1
    local name=$2
    local rlw=$3
    local w_alpha=$4
    local w_depth=$5
    local w_photo=$6
    local w_edge=$7
    local w_cov_align=$8
    local title=$9
    local desc=${10}

    cat > "${num}_${name}.yaml" << EOF
# Experiment ${num}: ${title}
# ${desc}

input_mesh_path: "assets/isosphere.obj"
target_mesh_path: "assets/spot.obj"
output_dir: "output/spot_exp/${num}_${name}/"

simulation:
  grid_dx: 1
  points_per_cell_cuberoot: 3
  grid_min_point: [-16.0, -16.0, -16.0]
  grid_max_point: [16.0, 16.0, 16.0]
  lam: 38888.89
  mu: 58333.3
  density: 75.0
  dt: 0.00833333333
  drag: 0.5
  external_force: [0.0, 0.0, 0.0]
  smoothing_factor: 0.955

optimization:
  num_animations: 60
  num_timesteps: 10
  control_stride: 10
  max_gd_iters: 10
  max_ls_iters: 15
  initial_alpha: 0.005
  gd_tol: 0.00005
  use_session_mode: false

  loss:
    enabled: true
    render_loss_weight: ${rlw}
    w_alpha: ${w_alpha}
    w_depth: ${w_depth}
    w_photo: ${w_photo}
    w_edge: ${w_edge}
    w_cov_align: ${w_cov_align}
    w_cov_reg: 0.01
    w_det_barrier: 0.1
    cov_reg_mode: 'eigenvalue'
    schedule: 'constant'

  magnitude_strategy: 'normalize'

upsample:
  use_simple_pipeline: true
  render_loss_weight: ${rlw}
  w_alpha: ${w_alpha}
  w_depth: ${w_depth}
  w_photo: ${w_photo}
  w_edge: ${w_edge}
  w_cov_align: ${w_cov_align}
  w_cov_reg: 0.01
  w_det_barrier: 0.1
  magnitude_strategy: 'normalize'

  covariance:
    use_curvature_for_target: true
    sigma_isotropic: 0.038
    curvature_sigma:
      sigma_n0: 0.02
      sigma_t0: 0.03
      a: 12.0
      b: 5.0
      a_n: 3
      u: 3
      k_floor_normal: 0.002
      k_floor_tangent: 0.005
    sigma0: 0.25
    k_F: 32
    use_multiscale_F: true
    k_F_coarse: 64
    k_F_fine: 16
    multiscale_blend_mode: 'adaptive'
    enable_subdivision: true
    subdivision_target: 60000
    subdivision_jitter: 0.08
    sv_min: 0.80
    sv_max: 1.20
    use_F_smoothing: true
    F_smooth:
      lambda_lap: 0.005
      num_nodes: 180
      node_knn: 8
      point_knn: 8

camera:
  width: 3840
  height: 2160
  fx: 1425.0
  fy: 1425.0
  cx: 1920.0
  cy: 1080.0
  znear: 0.01
  zfar: 100.0
  lookat:
    eye: [20.0, -25.0, 12.5]
    target: [0.0, 0.0, 0.0]
    up: [0.0, 0.0, 1.0]

render:
  num_frames: 1
  schedule: uniform
  bg: [1.0, 1.0, 1.0]
  training_resolution_scale: 1.0
  particle_color: [0.27, 0.51, 0.71]
  lighting:
    model: phong
    type: directional
    direction: [0.3, -0.5, 0.8]
    two_sided: true
    ambient: 0.18
    diffuse: 0.85
    specular: 0.10
    shininess: 32
EOF
    echo "✓ Created ${num}_${name}.yaml"
}

# Render weight variations
create_config "06" "render_1e3" "1e3" 2.0 5.0 1.0 3.0 10.0 \
    "Render Weight = 1e3 (Moderate)" \
    "Balanced render/physics influence"

create_config "07" "render_5e3" "5e3" 2.0 5.0 1.0 3.0 10.0 \
    "Render Weight = 5e3 (Strong)" \
    "Strong render influence"

create_config "08" "render_1e4" "1e4" 2.0 5.0 1.0 3.0 10.0 \
    "Render Weight = 1e4 (Very Strong)" \
    "Maximum render influence"

# Component weight variations
create_config "09" "edge_heavy" "1e4" 2.0 5.0 1.0 10.0 10.0 \
    "Edge-Heavy (w_edge=10.0)" \
    "Emphasize edge sharpness"

create_config "10" "depth_heavy" "1e4" 2.0 15.0 1.0 3.0 10.0 \
    "Depth-Heavy (w_depth=15.0)" \
    "Emphasize depth accuracy"

create_config "11" "cov_heavy" "1e4" 2.0 5.0 1.0 3.0 20.0 \
    "Covariance-Heavy (w_cov_align=20.0)" \
    "Emphasize F-gradient alignment"

create_config "12" "balanced" "1e4" 5.0 5.0 5.0 5.0 5.0 \
    "Balanced Components (All Equal)" \
    "Equal weight to all components"

echo ""
echo "================================================================"
echo "✓ All remaining configs created!"
echo "================================================================"
echo ""
echo "Total configs: 12"
echo "- 01-05: Already created"
echo "- 06-12: Just created"
echo ""
echo "Run experiments with:"
echo "  python run.py -c configs/spot_experiment/XX_*.yaml --png"
echo ""
echo "Batch run all:"
echo "  for c in configs/spot_experiment/0*.yaml; do"
echo "    python run.py -c \$c --png 2>&1 | tee logs/\$(basename \$c .yaml).log"
echo "  done"
echo ""
