#!/usr/bin/env python3
"""
Generate all ablation study configuration files for bunny and spot experiments.
"""

from pathlib import Path

# Base template
BASE_TEMPLATE = """# Experiment {exp_num}: {title}
# {description}

input_mesh_path: "assets/isosphere.obj"
target_mesh_path: "{target_mesh}"
output_dir: "output/ablation/{experiment}/{exp_name}/"

simulation:
  grid_dx: 1
  points_per_cell_cuberoot: 4
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
  num_animations: 20
  num_timesteps: 10
  control_stride: 1
  max_gd_iters: 1
  max_ls_iters: 15
  initial_alpha: 0.01
  gd_tol: 0.0
  use_session_mode: false
  adaptive_alpha_enabled: true

  loss:
    enabled: true
    w_alpha: {w_alpha}
    w_depth: {w_depth}
    w_photo: {w_photo}
    w_edge: {w_edge}
    w_cov_align: {w_cov_align}
    w_cov_reg: 0.01
    w_det_barrier: 0.1
    cov_reg_mode: 'eigenvalue'
    schedule: 'constant'

  magnitude_strategy: 'normalize'

upsample:
  use_simple_pipeline: true
  render_loss_weight: 1e4

  covariance:
    use_curvature_for_target: true
    sigma_isotropic: 0.038
    curvature_sigma:
      sigma_n0: 0.1
      sigma_t0: 0.12
      a: 6.0
      b: 5.0
      a_n: 1.5
      u: 1.5
      k_floor_normal: 0.002
      k_floor_tangent: 0.005
    sigma0: 0.12
    k_F: 32
    use_multiscale_F: true
    k_F_coarse: 64
    k_F_fine: 16
    multiscale_blend_mode: 'adaptive'
    enable_subdivision: false
    subdivision_target: 100000
    subdivision_jitter: 0.1
    sv_min: 0.80
    sv_max: 1.20
    use_F_smoothing: false
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
"""

# Experiment configurations
EXPERIMENTS = [
    {
        "exp_num": 3,
        "exp_name": "3_geometric_dominance",
        "title": "Geometric Dominance",
        "description": """Contribution: (w_depth * 1.75) ≈ 0.35 (Auxiliary)
#               (w_edge * 0.037) ≈ 2.0 (Dominant)
#               (w_cov_align * 0.006) ≈ 2.0 (Dominant)""",
        "weights": {
            "w_alpha": 0.0,
            "w_depth": 0.2,
            "w_photo": 0.0,
            "w_edge": 54.0,
            "w_cov_align": 330.0,
        }
    },
    {
        "exp_num": 4,
        "exp_name": "4_depth_led_edge_assisted",
        "title": "Depth-Led, Edge-Assisted",
        "description": """Contribution: (w_depth * 1.75) ≈ 5.25
#               (w_edge * 0.037) ≈ 5.0
#               (w_cov_align * 0.006) ≈ 0.96 (Auxiliary)""",
        "weights": {
            "w_alpha": 0.0,
            "w_depth": 3.0,
            "w_photo": 0.0,
            "w_edge": 135.0,
            "w_cov_align": 160.0,
        }
    },
    {
        "exp_num": 5,
        "exp_name": "5_alpha_led",
        "title": "Alpha-Led (Silhouette GPS)",
        "description": """Contribution: (w_alpha * 0.029) ≈ 1.0
#               (w_edge * 0.037) ≈ 1.0
#               (w_cov_align * 0.006) ≈ 0.96""",
        "weights": {
            "w_alpha": 35.0,
            "w_depth": 0.0,
            "w_photo": 0.0,
            "w_edge": 27.0,
            "w_cov_align": 160.0,
        }
    },
    {
        "exp_num": 6,
        "exp_name": "6_geometric_only_control",
        "title": "Geometric Only (Control - No GPS)",
        "description": """Contribution: (No directional signal)
# Sensors are on, but have no destination""",
        "weights": {
            "w_alpha": 0.0,
            "w_depth": 0.0,
            "w_photo": 0.0,
            "w_edge": 10.0,
            "w_cov_align": 20.0,
        }
    },
]

def generate_configs():
    """Generate all configuration files for bunny and spot."""

    experiments_info = {
        "bunny": {
            "target_mesh": "assets/bunny.obj"
        },
        "spot": {
            "target_mesh": "assets/spot.obj"
        }
    }

    for experiment_name, info in experiments_info.items():
        base_dir = Path(f"configs/ablation_study/{experiment_name}")
        base_dir.mkdir(parents=True, exist_ok=True)

        for exp in EXPERIMENTS:
            config_content = BASE_TEMPLATE.format(
                exp_num=exp["exp_num"],
                exp_name=exp["exp_name"],
                title=exp["title"],
                description=exp["description"],
                target_mesh=info["target_mesh"],
                experiment=experiment_name,
                **exp["weights"]
            )

            output_path = base_dir / f"{exp['exp_name']}.yaml"
            with open(output_path, 'w') as f:
                f.write(config_content)

            print(f"✅ Created: {output_path}")

if __name__ == "__main__":
    generate_configs()
    print("\n✅ All ablation study configurations generated!")
    print("\nGenerated files:")
    print("  configs/ablation_study/bunny/")
    print("    - 1_baseline.yaml (already created)")
    print("    - 2_equal_contribution.yaml (already created)")
    print("    - 3_geometric_dominance.yaml")
    print("    - 4_depth_led_edge_assisted.yaml")
    print("    - 5_alpha_led.yaml")
    print("    - 6_geometric_only_control.yaml")
    print("\n  configs/ablation_study/spot/")
    print("    - 1_baseline.yaml")
    print("    - 2_equal_contribution.yaml")
    print("    - 3_geometric_dominance.yaml")
    print("    - 4_depth_led_edge_assisted.yaml")
    print("    - 5_alpha_led.yaml")
    print("    - 6_geometric_only_control.yaml")
