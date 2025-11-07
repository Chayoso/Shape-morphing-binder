#!/usr/bin/env python
"""
Generate experimental configs for spot experiment suite
"""

BASE_TEMPLATE = """# Experiment {num}: {title}
# {description}

input_mesh_path: "assets/isosphere.obj"
target_mesh_path: "assets/spot.obj"
output_dir: "output/spot_exp/{num}_{name}/"

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
  initial_alpha: {alpha}
  gd_tol: 0.00005
  use_session_mode: false

  loss:
    enabled: {enabled}
    {loss_params}

  {magnitude_strategy}

upsample:
  use_simple_pipeline: true
  {upsample_params}

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
"""

# Experiments 05-08: Render weight variations
experiments = [
    # Render weight variations (alpha=0.005)
    {
        'num': '05',
        'name': 'render_1e2',
        'title': 'Render Weight = 1e2 (Weak Render)',
        'description': 'Minimal render influence',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e2',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 10.0
    },
    {
        'num': '06',
        'name': 'render_1e3',
        'title': 'Render Weight = 1e3 (Moderate Render)',
        'description': 'Balanced render influence',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e3',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 10.0
    },
    {
        'num': '07',
        'name': 'render_5e3',
        'title': 'Render Weight = 5e3 (Strong Render)',
        'description': 'High render influence',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '5e3',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 10.0
    },
    {
        'num': '08',
        'name': 'render_1e4',
        'title': 'Render Weight = 1e4 (Very Strong Render)',
        'description': 'Maximum render influence',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e4',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 10.0
    },
    # Component weight variations (render_weight=1e4)
    {
        'num': '09',
        'name': 'edge_heavy',
        'title': 'Edge-Heavy (w_edge=10.0)',
        'description': 'Emphasize edge sharpness',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e4',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 10.0, 'w_cov_align': 10.0
    },
    {
        'num': '10',
        'name': 'depth_heavy',
        'title': 'Depth-Heavy (w_depth=15.0)',
        'description': 'Emphasize depth accuracy',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e4',
        'w_alpha': 2.0, 'w_depth': 15.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 10.0
    },
    {
        'num': '11',
        'name': 'cov_heavy',
        'title': 'Covariance-Heavy (w_cov_align=20.0)',
        'description': 'Emphasize F-gradient alignment',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e4',
        'w_alpha': 2.0, 'w_depth': 5.0, 'w_photo': 1.0, 'w_edge': 3.0, 'w_cov_align': 20.0
    },
    {
        'num': '12',
        'name': 'balanced',
        'title': 'Balanced Components (All Equal)',
        'description': 'Equal weight to all components',
        'alpha': 0.005,
        'enabled': 'true',
        'render_weight': '1e4',
        'w_alpha': 5.0, 'w_depth': 5.0, 'w_photo': 5.0, 'w_edge': 5.0, 'w_cov_align': 5.0
    },
]

for exp in experiments:
    # Build loss params
    loss_params = f"""render_loss_weight: {exp['render_weight']}
    w_alpha: {exp['w_alpha']}
    w_depth: {exp['w_depth']}
    w_photo: {exp['w_photo']}
    w_edge: {exp['w_edge']}
    w_cov_align: {exp['w_cov_align']}
    w_cov_reg: 0.01
    w_det_barrier: 0.1
    cov_reg_mode: 'eigenvalue'
    schedule: 'constant'"""

    upsample_params = f"""render_loss_weight: {exp['render_weight']}
  w_alpha: {exp['w_alpha']}
  w_depth: {exp['w_depth']}
  w_photo: {exp['w_photo']}
  w_edge: {exp['w_edge']}
  w_cov_align: {exp['w_cov_align']}
  w_cov_reg: 0.01
  w_det_barrier: 0.1
  magnitude_strategy: 'normalize'"""

    magnitude_strategy = "magnitude_strategy: 'normalize'"

    content = BASE_TEMPLATE.format(
        num=exp['num'],
        title=exp['title'],
        description=exp['description'],
        name=exp['name'],
        alpha=exp['alpha'],
        enabled=exp['enabled'],
        loss_params=loss_params,
        magnitude_strategy=magnitude_strategy,
        upsample_params=upsample_params
    )

    filename = f"{exp['num']}_{exp['name']}.yaml"
    with open(filename, 'w') as f:
        f.write(content)

    print(f"✓ Created {filename}")

print(f"\nGenerated {len(experiments)} experiment configs!")
