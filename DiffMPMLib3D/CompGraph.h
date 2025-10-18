#pragma once
#include "pch.h"
#include "PointCloud.h"
#include "Grid.h"
#include <math.h>
#include <omp.h>
#include <iostream>
#include <vector>

namespace DiffMPMLib3D {

    // Represents the state of the simulation at a single timestep (or "layer").
    struct CompGraphLayer
    {
        std::shared_ptr<PointCloud> point_cloud = nullptr;
        std::shared_ptr<Grid> grid = nullptr;
    };
    
    // The main computational graph class that manages the simulation layers and runs the optimization process.
    class CompGraph
    {
    public:
        CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid);

        // Main entry point for running the optimization over multiple episodes.
        void OptimizeDefGradControlSequence(
            // SIMULATION PARAMS
            int num_steps, // number of timesteps, aka layers in the comp graph
            float _dt,
            float _drag,
            Vec3 _f_ext,
            // OPTIMIZATION PARAMS
            int control_stride,
            int max_gd_iters,
            int max_line_search_iters,
            float initial_alpha,
            float gd_tol,
            float smoothing_factor, 
            int current_episodes
        );

        // Sets up the computational graph by creating copies of the initial state for each layer.
        void SetUpCompGraph(int num_layers);

        // Computes the loss at the final layer based on mass distribution.
        float EndLayerMassLoss();

        // Runs the simulation forward from a given starting layer.
        void ComputeForwardPass(size_t start_layer);
        void ComputeForwardPass(size_t start_layer, int current_episode);
        
        // Runs the backpropagation process from the end of the graph to a given control layer.
        void ComputeBackwardPass(size_t control_layer);

        void OptimizeSingleTimestep(
            int timestep_idx,      // which timestep to optimize
            int max_gd_iters = 1,  // gradient descent iterations
            int current_episode = 0,
            float initial_alpha = 1.0f,  // initial step size
            int max_line_search_iters = 10  // line search iterations
        );

        // Gets the point cloud at a specific timestep.
        std::shared_ptr<PointCloud> GetPointCloudAtTimestep(int timestep_idx);

        // Utility for verifying gradients using finite differences (for debugging).
        void FiniteDifferencesGradientTest(int num_steps, size_t particle_id);

        // Stores all simulation states for each timestep.
        std::vector<CompGraphLayer> layers;
        // A read-only grid representing the target mass distribution.
        std::shared_ptr<const Grid> target_grid;
        std::vector<float> stored_render_grad_F_;  // Flattened (N*9,)
        std::vector<float> stored_render_grad_x_;  // Flattened (N*3,)
        bool has_render_grads_ = false;
        size_t render_grad_num_points_ = 0;
        
    private:
        // Simulation parameters cached for use in member functions.
        Vec3 f_ext = Vec3::Zero();
        float dt = 1.0f / 120.0f;
        float drag = 0.5f;
        float smoothing_factor = 0.1f;
        
        // Helper function to clip gradients to prevent explosions.
        static inline void ClipPointGradients(PointCloud& pc,
            float clip_dLdF = 5e-2f,
            float clip_dLdx = 1e-1f,
            float clip_dLdv = 1e-1f)
        {
    #pragma omp parallel for
            for (int i = 0; i < (int)pc.points.size(); ++i) {
                auto& pt = pc.points[i];
                float nf = pt.dLdF.norm();
                if (nf > clip_dLdF) pt.dLdF *= (clip_dLdF / std::max(nf, 1e-12f));
                // Similar clipping for dLdx and dLdv can be added here if needed.
            }
        }
    };
}