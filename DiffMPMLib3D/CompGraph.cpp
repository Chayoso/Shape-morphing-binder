#include "pch.h"
#include "CompGraph.h"
#include "ForwardSimulation.h"
#include "BackPropagation.h"
#include "Interpolation.h"
#include <cmath>
#include <numeric> 
#include <fstream>
#include <filesystem>

#include <algorithm>
#ifdef DIAGNOSTICS
#include <mutex>
#endif

namespace DiffMPMLib3D {

    // Use a 'using namespace' directive to simplify function calls within this .cpp file.
    using namespace SingleThreadMPM;

    CompGraph::CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid)
    {
        layers.clear();
        layers.resize(1);
        layers[0].point_cloud = std::make_shared<PointCloud>(*initial_point_cloud);
        layers[0].grid = std::make_shared<Grid>(*grid);
        target_grid = _target_grid;
    }

    void CompGraph::SetUpCompGraph(int num_layers)
    {
        assert(num_layers > 0);
        layers.resize(num_layers);

    #pragma omp parallel for 
        for (int i = 1; i < num_layers; i++) {
            layers[i].point_cloud = std::make_shared<PointCloud>(*layers.front().point_cloud);
            layers[i].grid = std::make_shared<Grid>(*layers.front().grid);
        }
    }

   float CompGraph::EndLayerMassLoss()
    {
        float out_of_target_penalty = 5.f;
        float eps = 1e-4f;
        float min_mass = 1e-3f;
        float penalty_weight = 1.f;

        PointCloud& point_cloud = *layers.back().point_cloud;
        Grid& grid = *layers.back().grid;
        float dx = grid.dx;

        G_Reset(grid);
        P2G(point_cloud, grid, 0.f, 0.f);

        float loss = 0.f;
        int dim_x = target_grid->dim_x;
        int dim_y = target_grid->dim_y;
        int dim_z = target_grid->dim_z;

    #pragma omp parallel for reduction(+:loss)
        for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
            int i = idx / (dim_y * dim_z);
            int j = (idx / dim_z) % dim_y;
            int k = idx % dim_z;

            float c_m = grid.GetNode(i, j, k).m;
            float t_m = target_grid->GetNode(i, j, k).m;

            float log_diff = std::log(c_m + 1.f + eps) - std::log(t_m + 1.f + eps);
            loss += 0.5f * log_diff * log_diff;
            // Chain rule: d/dc_m [log(c_m+1)] = 1 / (c_m+1)
            grid.GetNode(i, j, k).dLdm = log_diff / (c_m + 1.f + eps);

            if (c_m < min_mass) {
                float diff = min_mass - c_m;
                loss += penalty_weight * diff * diff;
                grid.GetNode(i, j, k).dLdm += -2.f * penalty_weight * diff;
            }
        }

    #pragma omp parallel for
        for (int p = 0; p < point_cloud.points.size(); p++) {
            MaterialPoint& mp = point_cloud.points[p];
            mp.dLdx.setZero();

            std::vector<std::array<int, 3>> indices;
            auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);

            for (int i = 0; i < (int)nodes.size(); i++) {
                const GridNode& node = nodes[i];
                Vec3 dgp = node.x - mp.x;
                
                Vec3 dgp_div_dx = dgp / dx;
                Vec3 bspline_vals(CubicBSpline(dgp_div_dx[0]), CubicBSpline(dgp_div_dx[1]), CubicBSpline(dgp_div_dx[2]));
                Vec3 bspline_slopes(CubicBSplineSlope(dgp_div_dx[0]), CubicBSplineSlope(dgp_div_dx[1]), CubicBSplineSlope(dgp_div_dx[2]));
                
                Vec3 wgpGrad = -1.f / dx * Vec3(
                    bspline_slopes[0] * bspline_vals[1] * bspline_vals[2],
                    bspline_vals[0] * bspline_slopes[1] * bspline_vals[2],
                    bspline_vals[0] * bspline_vals[1] * bspline_slopes[2]
                );

                const auto& target_node = target_grid->GetNode(indices[i][0], indices[i][1], indices[i][2]);
                float penalty = (target_node.m > 1e-12f) ? 1.0f : out_of_target_penalty;
                mp.dLdx += penalty * mp.m * node.dLdm * wgpGrad;
            }

            mp.dLdF.setZero();
            mp.dLdv.setZero();
            mp.dLdC.setZero();
        }

        return loss;
    }

    void CompGraph::ComputeForwardPass(size_t start_layer, int current_episode)
    {
        for (size_t i = start_layer; i < layers.size() - 1; i++)
        {
            ForwardTimeStep(
                *layers[i + 1].point_cloud,
                *layers[i].point_cloud,
                *layers[i].grid,
                smoothing_factor, dt, drag, f_ext, current_episode);
        }
    }

    void CompGraph::ComputeBackwardPass(size_t control_layer)
    {
        // ════════════════════════════════════════════════════════════════════════
        // [FIX] GRADIENT INJECTION FIX: Add render gradients to final layer
        // [CRITICAL] Only inject ONCE per pass using flag check!
        // ════════════════════════════════════════════════════════════════════════
        if (has_render_grads_ && render_grad_num_points_ > 0 && !render_grads_injected_this_control_timestep_) {
            std::cout << "[Backward] [FIX] Injecting render gradients into final layer (ONCE per pass)..." << std::endl;

            // Get final layer
            size_t final_layer = layers.size() - 1;
            auto& final_pc = layers[final_layer].point_cloud;

            // Verify sizes match
            size_t num_points = std::min(final_pc->points.size(), render_grad_num_points_);

            if (final_pc->points.size() != render_grad_num_points_) {
                std::cerr << "[WARN]  WARNING: Point count mismatch! "
                          << "Expected: " << render_grad_num_points_
                          << ", Got: " << final_pc->points.size() << std::endl;
            }

            // Inject render gradients into physics gradients
            #pragma omp parallel for
            for (size_t i = 0; i < num_points; ++i) {
                auto& pt = final_pc->points[i];

                // Add F gradients (deformation gradient: 3x3 matrix = 9 components)
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        int idx = i * 9 + r * 3 + c;
                        pt.dLdF(r, c) += render_gain_ * stored_render_grad_F_[idx];
                    }
                }

                // Add x gradients (position: 3D vector = 3 components)
                for (int d = 0; d < 3; d++) {
                    int idx = i * 3 + d;
                    pt.dLdx[d] += render_gain_ * stored_render_grad_x_[idx];
                }
            }

            // Diagnostic output
            double total_dLdF_norm = 0.0;
            double total_dLdx_norm = 0.0;

            #pragma omp parallel for reduction(+:total_dLdF_norm, total_dLdx_norm)
            for (size_t i = 0; i < num_points; ++i) {
                total_dLdF_norm += final_pc->points[i].dLdF.norm();
                total_dLdx_norm += final_pc->points[i].dLdx.norm();
            }

            std::cout << "[Backward] [OK] Render gradients injected successfully!" << std::endl;
            std::cout << "  ├─ Particles: " << num_points << std::endl;
            std::cout << "  ├─ Render gain: " << render_gain_ << std::endl;
            std::cout << "  ├─ ||∂L/∂F|| total: " << total_dLdF_norm << std::endl;
            std::cout << "  └─ ||∂L/∂x|| total: " << total_dLdx_norm << std::endl;

            // Mark as injected to prevent re-injection
            render_grads_injected_this_control_timestep_ = true;
        }
        // ════════════════════════════════════════════════════════════════════════

        for (int i = (int)layers.size() - 2; i >= (int)control_layer; i--)
        {
            layers[i].grid->ResetGradients();
            layers[i].point_cloud->ResetGradients();
            
            // [OK] This will propagate BOTH physics AND render gradients backward
            // because we already injected render grads to the last layer above
            Back_Timestep(layers[i + 1], layers[i], drag, dt, smoothing_factor);
            
            // [FIX] NEW: Apply physics_weight to balance physics/render signals
            // [WARN]  Only apply if render gradients are present (E2E mode)
            if (physics_weight_ != 1.0f && has_render_grads_) {
                auto& pc = *layers[i].point_cloud;
                auto& grid = *layers[i].grid;

                // Scale point cloud gradients
                #pragma omp parallel for
                for (int p = 0; p < (int)pc.points.size(); ++p) {
                    auto& mp = pc.points[p];
                    mp.dLdF *= physics_weight_;
                    mp.dLdx *= physics_weight_;
                    mp.dLdv *= physics_weight_;
                    mp.dLdC *= physics_weight_;
                }

                // Scale grid gradients
                #pragma omp parallel for
                for (int idx = 0; idx < grid.dim_x * grid.dim_y * grid.dim_z; ++idx) {
                    int ii = idx / (grid.dim_y * grid.dim_z);
                    int jj = (idx / grid.dim_z) % grid.dim_y;
                    int kk = idx % grid.dim_z;
                    auto& node = grid.GetNode(ii, jj, kk);
                    node.dLdv *= physics_weight_;
                    node.dLdm *= physics_weight_;
                    node.dLdp *= physics_weight_;
                }
            }
        }

    }

    void CompGraph::OptimizeDefGradControlSequence(
        int num_steps, float _dt, float _drag, Vec3 _f_ext,
        int control_stride, int max_gd_iters, int max_line_search_iters,
        float initial_alpha, float gd_tol, float _smoothing_factor, int current_episodes,
        bool adaptive_alpha_enabled, float adaptive_alpha_target_norm, float adaptive_alpha_min_scale,
        bool skip_setup)
    {
        dt = _dt;
        drag = _drag;
        smoothing_factor = _smoothing_factor;
        f_ext = _f_ext;

        // ═══════════════════════════════════════════════════════════════════════
        // [CONFIG] OPTIMIZATION CONFIGURATION SUMMARY
        // ═══════════════════════════════════════════════════════════════════════
        std::cout << "\n" << std::string(75, '=') << std::endl;
        std::cout << "[CONFIG] PHYSICS OPTIMIZATION CONFIGURATION" << std::endl;
        std::cout << std::string(75, '=') << std::endl;

        // Simulation Parameters
        std::cout << "[INFO] Simulation Parameters:" << std::endl;
        std::cout << "  ├─ Timesteps:        " << num_steps << std::endl;
        std::cout << "  ├─ dt:               " << dt << std::endl;
        std::cout << "  ├─ Drag:             " << drag << std::endl;
        std::cout << "  ├─ Smoothing factor: " << smoothing_factor << std::endl;
        std::cout << "  └─ External force:   [" << f_ext[0] << ", " << f_ext[1] << ", " << f_ext[2] << "]" << std::endl;

        // Optimization Parameters
        std::cout << "\n[OPT]  Optimization Parameters:" << std::endl;
        std::cout << "  ├─ Control stride:   " << control_stride << std::endl;
        std::cout << "  ├─ Max GD iters:     " << max_gd_iters << std::endl;
        std::cout << "  ├─ Max LS iters:     " << max_line_search_iters << std::endl;
        std::cout << "  ├─ Initial alpha:    " << initial_alpha << std::endl;
        std::cout << "  ├─ GD tolerance:     " << gd_tol << std::endl;
        std::cout << "  └─ Adaptive alpha:   " << (adaptive_alpha_enabled ? "ENABLED" : "DISABLED") << std::endl;

        if (adaptive_alpha_enabled) {
            std::cout << "      ├─ Target norm:  " << adaptive_alpha_target_norm << std::endl;
            std::cout << "      └─ Min scale:    " << adaptive_alpha_min_scale << std::endl;
        }

        // Adam Optimizer Parameters (hardcoded)
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-3f;
        std::cout << "\n[ADAM] Adam Optimizer:" << std::endl;
        std::cout << "  ├─ beta1:            " << beta1 << std::endl;
        std::cout << "  ├─ beta2:            " << beta2 << std::endl;
        std::cout << "  └─ epsilon:          " << epsilon << std::endl;

        std::cout << std::string(75, '=') << "\n" << std::endl;

        // [FIX] MULTI-PASS FIX: Only setup on first pass
        if (!skip_setup) {
            std::cout << "[Setup] Initializing computation graph..." << std::endl;
            SetUpCompGraph(num_steps);
            // Reset Adam optimizer state for fresh optimization
            adam_timestep_ = 0;
            std::cout << "[Adam] Reset adam_timestep to 0 (fresh optimization)" << std::endl;
        } else {
            std::cout << "[Setup] Skipping SetUpCompGraph (using existing state from previous pass)" << std::endl;
            std::cout << "[Adam] Continuing from adam_timestep = " << adam_timestep_ << " (preserving momentum)" << std::endl;
        }

        // [FIX] CRITICAL FIX: Reset render gradient injection flag ONCE per pass
        // Must be BEFORE any ComputeBackwardPass calls to prevent double injection
        render_grads_injected_this_control_timestep_ = false;

        ComputeForwardPass(0, current_episodes);

        auto eval_loss = [&]() -> float { return EndLayerMassLoss(); };
        float initial_loss = eval_loss();
        std::cout << "Initial loss = " << initial_loss << std::endl;

        ComputeBackwardPass(0);
        float initial_norm_global = layers.front().point_cloud->Compute_dLdF_Norm();
        std::cout << "Initial global gradient norm = " << initial_norm_global << std::endl;

        size_t num_points = layers.front().point_cloud->points.size();
        std::vector<Mat3> dFc_bak(num_points), m_bak(num_points), v_bak(num_points), vmax_bak(num_points);

        // =======================================================================
        // Single-pass optimization (multipass handled in Python E2E loop)
        // Python controls Pass 1/2/3, C++ does one optimization pass per call
        // =======================================================================
        int totalTemporalIterations = 1;
        std::cout << "Starting optimization (Episode " << current_episodes << ")" << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        for (int temporalIter = 0; temporalIter < totalTemporalIterations; ++temporalIter)
        {
            int num_control_steps = (num_steps - 1 + control_stride - 1) / control_stride;
            for (int control_timestep = 0; control_timestep < num_steps - 1; control_timestep += control_stride)
            {
                int step_number = control_timestep / control_stride + 1;
                std::cout << "\n[STEP] Control Step " << step_number << "/" << num_control_steps
                          << " (Timestep " << control_timestep << ")" << std::endl;

                // [REMOVED] DON'T reset flag here - it causes render grads to inject 10x!

                // [FIX] ADAPTIVE INITIAL_ALPHA: Reduce alpha when gradients are too large
                float alpha = initial_alpha;

                if (adaptive_alpha_enabled) {
                    // Compute current gradient norm
                    ComputeBackwardPass(control_timestep);
                    float current_grad_norm = layers.front().point_cloud->Compute_dLdF_Norm();

                    // Compute adaptive alpha: reduce when gradients are larger than target
                    float alpha_scale = std::min(1.0f, adaptive_alpha_target_norm / std::max(current_grad_norm, 1e-6f));
                    alpha_scale = std::max(alpha_scale, adaptive_alpha_min_scale);  // Clamp to minimum

                    alpha = initial_alpha * alpha_scale;

                    // Print adaptive alpha info
                    if (alpha_scale < 1.0f) {
                        std::cout << "  [Adaptive Alpha] grad_norm=" << current_grad_norm
                                  << ", target=" << adaptive_alpha_target_norm
                                  << ", scale=" << alpha_scale
                                  << ", alpha=" << alpha << " (reduced from " << initial_alpha << ")" << std::endl;
                    }
                } else {
                    // Fixed alpha mode: still need backward pass for gradient
                    ComputeBackwardPass(control_timestep);
                    std::cout << "  [Fixed Alpha] alpha=" << alpha << " (adaptive disabled)" << std::endl;
                }

                float initial_norm_local = 0.f;

                for (int gd_iter = 0; gd_iter < max_gd_iters; ++gd_iter)
                {
                    // [DEBUG] Track Adam timestep for momentum debugging
                    if (gd_iter == 0) {
                        std::cout << "  [Adam] Current adam_timestep = " << adam_timestep_
                                  << " (next update will use timestep " << (adam_timestep_ + 1) << ")" << std::endl;
                    }

                    ComputeForwardPass(control_timestep, current_episodes);
                    float gd_loss = eval_loss();

#ifdef DIAGNOSTICS
                    // --- DIAG: Log J statistics for the current forward state
                    {
                        static std::once_flag header_once;
                        static std::mutex io_mtx;
                        auto& pc_last = *layers.back().point_cloud;
                        std::vector<float> J = pc_last.GetPointDeterminants(); // det(F + dFc)
                        if (!J.empty()) {
                            std::vector<float> JJ = J; std::sort(JJ.begin(), JJ.end());
                            auto P = [&](float q){ size_t i = (size_t)std::floor(q * (JJ.size()-1));
                                                   return JJ[std::min(i, JJ.size()-1)]; };
                            const float j_min = JJ.front();
                            const float j_mean = std::accumulate(J.begin(), J.end(), 0.f) / float(J.size());
                            std::ofstream ofs("diag_opt.csv", std::ios::app);
                            std::call_once(header_once, [&](){
                                ofs << "pass,step,gd_iter,phase,loss,j_min,j_mean,j_p01,j_p50,j_p99,alpha_try,ls_iters,accepted\n";
                            });
                            ofs << temporalIter << "," << control_timestep << "," << gd_iter
                                << ",pre_ls," << gd_loss << ","
                                << j_min << "," << j_mean << "," << P(0.01f) << "," << P(0.50f) << "," << P(0.99f)
                                << "," << initial_alpha << "," << 0 << ",-1\n";
                        }
                    }
#endif

                    if (!std::isfinite(gd_loss)) {
                        std::cout << "Warning: Non-finite loss detected. Aborting step." << std::endl;
                        break;
                    }

                    ComputeBackwardPass(control_timestep);

                    auto& pc = *layers[control_timestep].point_cloud;
                    float gradient_norm = std::max(pc.Compute_dLdF_Norm(), 1e-12f);

                    if (gd_iter == 0) {
                        initial_norm_local = gradient_norm;
                    }

                    if (gradient_norm < gd_tol * initial_norm_local) {
                        std::cout << "Converged at GD iteration " << gd_iter << "." << std::endl;
                        break;
                    }

                #pragma omp parallel for
                    for (int i = 0; i < num_points; ++i) {
                        dFc_bak[i] = pc.points[i].dFc;
                        m_bak[i] = pc.points[i].momentum;
                        v_bak[i] = pc.points[i].vector;
                        vmax_bak[i] = pc.points[i].vector_max;
                    }

                    bool step_accepted = false;
                    float alpha_try = alpha;

                    for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter)
                    {
                        pc.Descend_Adam(alpha_try, gradient_norm, beta1, beta2, epsilon, adam_timestep_ + 1);
                        ComputeForwardPass(control_timestep, current_episodes);
                        float new_loss = eval_loss();

                        if (std::isfinite(new_loss) && new_loss < gd_loss) {
                            adam_timestep_++;
                            std::cout << "    [Adam] Step accepted! adam_timestep incremented: "
                                      << (adam_timestep_ - 1) << " → " << adam_timestep_ << std::endl;
                            alpha = std::min(alpha_try * 1.1f, initial_alpha);
                            step_accepted = true;
#ifdef DIAGNOSTICS
                            // mark acceptance
                            std::ofstream ofs("diag_opt.csv", std::ios::app);
                            ofs << temporalIter << "," << control_timestep << "," << gd_iter
                                << ",ls_accept," << new_loss << ",,,,,,"
                                << "," << alpha << "," << 0 << "," << 1 << "\n";
#endif
                            
                            break;
                        }

                    #pragma omp parallel for
                        for (int i = 0; i < num_points; ++i) {
                            pc.points[i].dFc = dFc_bak[i];
                            pc.points[i].momentum = m_bak[i];
                            pc.points[i].vector = v_bak[i];
                            pc.points[i].vector_max = vmax_bak[i];
                        }
                        alpha_try *= 0.5f;
                    }

                    if (!step_accepted) {
                        std::cout << "Line search failed. Moving to next control step." << std::endl;

#ifdef DIAGNOSTICS
                        std::ofstream ofs("diag_opt.csv", std::ios::app);
                        ofs << temporalIter << "," << control_timestep << "," << gd_iter
                            << ",ls_fail," << gd_loss << ",,,,,,"
                            << "," << alpha << "," << max_line_search_iters << "," << 0 << "\n";
#endif
                        break;
                    }
                } // End gradient descent loop
            } // End control timestep loop
        } // --- End temporalIter (multipass) loop ---
        
        float final_loss = eval_loss();
        std::cout << "\n" << std::string(75, '=') << std::endl;
        std::cout << "[OK] Optimization Complete " << std::endl;
        std::cout << std::string(75, '=') << std::endl;
        std::cout << "  Initial loss: " << initial_loss << std::endl;
        std::cout << "  Final loss:   " << final_loss << std::endl;
        std::cout << "  Reduction:    " << (initial_loss - final_loss) << " ("
                  << (100.0f * (initial_loss - final_loss) / std::max(initial_loss, 1e-6f)) << "%)" << std::endl;
        std::cout << std::string(75, '=') << "\n" << std::endl;
    }
    void CompGraph::OptimizeSingleTimestep(
        int timestep_idx,
        int max_gd_iters,
        int current_episode,
        float initial_alpha,
        int max_line_search_iters)
    {
        // Validate index
        if (timestep_idx < 0 || timestep_idx >= (int)layers.size() - 1) {
            std::cerr << "Invalid timestep index: " << timestep_idx << std::endl;
            return;
        }
        
        std::cout << "\n[Single Timestep Optimization]" << std::endl;
        std::cout << "  ├─ Timestep:      " << timestep_idx << std::endl;
        std::cout << "  ├─ Initial alpha: " << initial_alpha << std::endl;
        std::cout << "  ├─ Max GD iters:  " << max_gd_iters << std::endl;
        std::cout << "  └─ Max LS iters:  " << max_line_search_iters << std::endl;
        
        auto& pc = *layers[timestep_idx].point_cloud;
        size_t num_points = pc.points.size();
        
        // Backup for line search
        std::vector<Mat3> dFc_bak(num_points);
        std::vector<Mat3> m_bak(num_points);
        std::vector<Mat3> v_bak(num_points);
        std::vector<Mat3> vmax_bak(num_points);
        
        // Adam parameters
        const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-3f;
        // NOTE: Using member variable adam_timestep_ (persistent across calls)
        
        float alpha = initial_alpha;
        
        // Gradient descent loop
        for (int gd_iter = 0; gd_iter < max_gd_iters; ++gd_iter)
        {
            // 1. Forward pass from this timestep
            ComputeForwardPass(timestep_idx, current_episode);
            float gd_loss = EndLayerMassLoss();
            
            if (!std::isfinite(gd_loss)) {
                std::cout << "Warning: Non-finite loss. Aborting." << std::endl;
                break;
            }
            
            // 2. Backward pass to this timestep
            ComputeBackwardPass(timestep_idx);
            
            float gradient_norm = std::max(pc.Compute_dLdF_Norm(), 1e-12f);
            
            // 3. Backup current state
            #pragma omp parallel for
            for (int i = 0; i < (int)num_points; ++i) {
                dFc_bak[i] = pc.points[i].dFc;
                m_bak[i] = pc.points[i].momentum;
                v_bak[i] = pc.points[i].vector;
                vmax_bak[i] = pc.points[i].vector_max;
            }
            
            // 4. Line search
            bool step_accepted = false;
            float alpha_try = alpha;
            
            for (int ls_iter = 0; ls_iter < max_line_search_iters; ++ls_iter)
            {
                // Try Adam update
                pc.Descend_Adam(alpha_try, gradient_norm, beta1, beta2, epsilon, adam_timestep_ + 1);
                
                // Forward pass to evaluate new loss
                ComputeForwardPass(timestep_idx, current_episode);
                float new_loss = EndLayerMassLoss();
                
                if (std::isfinite(new_loss) && new_loss < gd_loss) {
                    // Accept step
                    adam_timestep_++;
                    alpha = std::min(alpha_try * 1.1f, initial_alpha);
                    step_accepted = true;
                    std::cout << "  Step accepted at ls_iter=" << ls_iter 
                              << ", loss=" << new_loss << std::endl;
                    break;
                }
                
                // Restore state and reduce step size
                #pragma omp parallel for
                for (int i = 0; i < (int)num_points; ++i) {
                    pc.points[i].dFc = dFc_bak[i];
                    pc.points[i].momentum = m_bak[i];
                    pc.points[i].vector = v_bak[i];
                    pc.points[i].vector_max = vmax_bak[i];
                }
                alpha_try *= 0.5f;
            }
            
            if (!step_accepted) {
                std::cout << "  Line search failed." << std::endl;
                break;
            }
        }
    }
    
    std::shared_ptr<PointCloud> CompGraph::GetPointCloudAtTimestep(int timestep_idx)
    {
        if (timestep_idx < 0 || timestep_idx >= (int)layers.size()) {
            std::cerr << "Invalid timestep index: " << timestep_idx << std::endl;
            return nullptr;
        }
        return layers[timestep_idx].point_cloud;
    }
    
    // ============================================================================
    // [FIX] NEW: Gradient Norm Monitoring
    // ============================================================================
    
    std::pair<double, double> CompGraph::GetLastLayerPhysGradNorm() const {
        if (layers.empty() || !layers.back().point_cloud) {
            return {0.0, 0.0};
        }
        
        const auto& pc = *layers.back().point_cloud;
        double gF2 = 0.0, gx2 = 0.0;
        
        #pragma omp parallel for reduction(+:gF2,gx2)
        for (int i = 0; i < (int)pc.points.size(); ++i) {
            const auto& p = pc.points[i];
            
            // dLdF Frobenius norm
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double v = p.dLdF(r, c);
                    gF2 += v * v;
                }
            }
            
            // dLdx L2 norm
            for (int k = 0; k < 3; ++k) {
                double v = p.dLdx(k);
                gx2 += v * v;
            }
        }
        
        return {std::sqrt(gF2), std::sqrt(gx2)};
    }
    
    std::pair<double, double> CompGraph::GetLayerPhysGradNorm(int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= (int)layers.size() || !layers[layer_idx].point_cloud) {
            return {0.0, 0.0};
        }

        const auto& pc = *layers[layer_idx].point_cloud;
        double gF2 = 0.0, gx2 = 0.0;

        #pragma omp parallel for reduction(+:gF2,gx2)
        for (int i = 0; i < (int)pc.points.size(); ++i) {
            const auto& p = pc.points[i];

            // dLdF Frobenius norm
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    double v = p.dLdF(r, c);
                    gF2 += v * v;
                }
            }

            // dLdx L2 norm
            for (int k = 0; k < 3; ++k) {
                double v = p.dLdx(k);
                gx2 += v * v;
            }
        }

        return {std::sqrt(gF2), std::sqrt(gx2)};
    }

    // [FIX] NEW: Get actual physics gradients for PCGrad
    std::pair<std::vector<Mat3>, std::vector<Vec3>> CompGraph::GetLastLayerPhysGradients() const {
        if (layers.empty() || !layers.back().point_cloud) {
            return {{}, {}};
        }

        const auto& pc = *layers.back().point_cloud;
        return {pc.GetPointDefGradGradients(), pc.GetPointPositionGradients()};
    }
}
