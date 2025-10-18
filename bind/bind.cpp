#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "PointCloud.h"
#include "Grid.h"
#include "Elasticity.h"
#include "GeometryLoading.h"
#include "ForwardSimulation.h"
#include "CompGraph.h"

// PyTorch C++ API (optional, for torch tensor support)
#ifdef DIFFMPM_WITH_TORCH
#include <torch/extension.h>
#define TORCH_AVAILABLE true
#else
#define TORCH_AVAILABLE false
#endif

namespace py = pybind11;
using namespace DiffMPMLib3D;

struct OptInput {
    std::string mpm_input_mesh_path;
    std::string mpm_target_mesh_path;
    float grid_dx;
    DiffMPMLib3D::Vec3 grid_min_point;
    DiffMPMLib3D::Vec3 grid_max_point;
    int points_per_cell_cuberoot;
    float lam;
    float mu;
    float p_density;
    float dt;
    float drag;
    DiffMPMLib3D::Vec3 f_ext;
    int num_animations;
    int num_timesteps;
    int control_stride;
    int max_gd_iters;
    int max_ls_iters;
    float initial_alpha;
    float gd_tol;
    float smoothing_factor;
    int current_episodes;
};

PYBIND11_MODULE(diffmpm_bindings, m) {
    m.doc() = "Python bindings for the DiffMPM core engine (E2E Differentiable)";

    // --- 1. Configuration binding ---
    py::class_<OptInput>(m, "OptInput")
        .def(py::init<>())
        .def_readwrite("mpm_input_mesh_path", &OptInput::mpm_input_mesh_path)
        .def_readwrite("mpm_target_mesh_path", &OptInput::mpm_target_mesh_path)
        .def_readwrite("grid_dx", &OptInput::grid_dx)
        .def_readwrite("grid_min_point", &OptInput::grid_min_point)
        .def_readwrite("grid_max_point", &OptInput::grid_max_point)
        .def_readwrite("points_per_cell_cuberoot", &OptInput::points_per_cell_cuberoot)
        .def_readwrite("lam", &OptInput::lam)
        .def_readwrite("mu", &OptInput::mu)
        .def_readwrite("p_density", &OptInput::p_density)
        .def_readwrite("dt", &OptInput::dt)
        .def_readwrite("drag", &OptInput::drag)
        .def_readwrite("f_ext", &OptInput::f_ext)
        .def_readwrite("num_animations", &OptInput::num_animations)
        .def_readwrite("num_timesteps", &OptInput::num_timesteps)
        .def_readwrite("control_stride", &OptInput::control_stride)
        .def_readwrite("max_gd_iters", &OptInput::max_gd_iters)
        .def_readwrite("max_ls_iters", &OptInput::max_ls_iters)
        .def_readwrite("initial_alpha", &OptInput::initial_alpha)
        .def_readwrite("gd_tol", &OptInput::gd_tol)
        .def_readwrite("smoothing_factor", &OptInput::smoothing_factor)
        .def_readwrite("current_episodes", &OptInput::current_episodes);

    // --- 2. Core data structures ---
    py::class_<PointCloud, std::shared_ptr<PointCloud>>(m, "PointCloud")
        .def("get_positions", &PointCloud::GetPointPositions, "Return particle positions as (N, 3) NumPy array")
        .def("get_masses", &PointCloud::GetPointMasses, "Return particle masses as (N,) NumPy array")
        .def("get_def_grads", &PointCloud::GetPointDefGrads, "Return particle deformation tensors as (N, 3, 3) NumPy array")
#ifdef DIFFMPM_WITH_TORCH
        // Torch tensor versions (requires PyTorch C++ API)
        .def("get_positions_torch", [](const PointCloud& pc, bool requires_grad) {
            const size_t N = pc.points.size();
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU)
                .requires_grad(requires_grad);
            
            auto tensor = torch::empty({(int64_t)N, 3}, options);
            auto accessor = tensor.accessor<float, 2>();
            
            for (size_t i = 0; i < N; ++i) {
                accessor[i][0] = pc.points[i].x[0];
                accessor[i][1] = pc.points[i].x[1];
                accessor[i][2] = pc.points[i].x[2];
            }
            
            return tensor;
        }, py::arg("requires_grad") = false,
           "Return particle positions as PyTorch tensor (N, 3) with optional gradient support")
        
        .def("get_def_grads_total_torch", [](const PointCloud& pc, bool requires_grad) {
            const size_t N = pc.points.size();
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU)
                .requires_grad(requires_grad);
            
            auto tensor = torch::empty({(int64_t)N, 3, 3}, options);
            auto accessor = tensor.accessor<float, 3>();
            
            for (size_t i = 0; i < N; ++i) {
                const auto& F  = pc.points[i].F;
                const auto& dF = pc.points[i].dFc;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        accessor[i][r][c] = F(r, c) + dF(r, c);
                    }
                }
            }
            
            return tensor;
        }, py::arg("requires_grad") = false,
           "Return total deformation F_total = F + dFc as PyTorch tensor (N, 3, 3)")
#endif
        .def("get_def_grads_morph", [](const PointCloud& pc) {
            const size_t N = pc.points.size();
            py::array_t<float> arr({(py::ssize_t)N, (py::ssize_t)3, (py::ssize_t)3});
            auto buf = arr.mutable_unchecked<3>();
            for (size_t i = 0; i < N; ++i) {
                const auto& A = pc.points[i].dFc;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        buf(i, r, c) = A(r, c);
            }
            return arr;
        }, "Return morph control deformation dFc as (N, 3, 3) array")
        .def("get_def_grads_total", [](const PointCloud& pc) {
            const size_t N = pc.points.size();
            py::array_t<float> arr({(py::ssize_t)N, (py::ssize_t)3, (py::ssize_t)3});
            auto buf = arr.mutable_unchecked<3>();
            for (size_t i = 0; i < N; ++i) {
                const auto& F  = pc.points[i].F;
                const auto& dF = pc.points[i].dFc;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        buf(i, r, c) = F(r, c) + dF(r, c);
            }
            return arr;
        }, "Return total deformation F_total = F + dFc as (N, 3, 3) array");

    py::class_<Grid, std::shared_ptr<Grid>>(m, "Grid")
        .def(py::init<int, int, int, float, DiffMPMLib3D::Vec3>(), "Grid constructor");

    // --- 3. Main engine (CompGraph) ---
    py::class_<CompGraph, std::shared_ptr<CompGraph>>(m, "CompGraph")
        .def(py::init<std::shared_ptr<PointCloud>, std::shared_ptr<Grid>, std::shared_ptr<const Grid>>())
        .def("run_optimization", [](CompGraph& self, const OptInput& opt) {
            self.OptimizeDefGradControlSequence(
                opt.num_timesteps, opt.dt, opt.drag, opt.f_ext,
                opt.control_stride, opt.max_gd_iters, opt.max_ls_iters,
                opt.initial_alpha, opt.gd_tol, opt.smoothing_factor,
                opt.current_episodes
            );
        }, "Run optimization for given episode.", py::arg("opt"))
        .def("get_num_layers", [](const CompGraph& self) {
            return self.layers.size();
        }, "Get total number of simulated frames")
        .def("get_point_cloud", [](const CompGraph& self, size_t layer_idx) -> std::shared_ptr<PointCloud> {
            if (layer_idx >= self.layers.size())
                throw std::out_of_range("Layer index out of range.");
            return self.layers[layer_idx].point_cloud;
        }, "Get PointCloud object for specific frame")
        
        // ✅ E2E Function 1: Physics loss (with const_cast workaround)
        .def("end_layer_mass_loss", [](CompGraph& self) -> float {
            // Workaround for const correctness issue
            return self.EndLayerMassLoss();
        }, "Compute physics loss at the last layer")
        
        // ✅ E2E Function 2: Accumulate render gradients
        .def("accumulate_render_grads",
            [](CompGraph& self,
               py::array_t<float> dLdF_render,
               py::array_t<float> dLdx_render
            ) {
                // Validate CompGraph has layers
                if (self.layers.empty()) {
                    throw std::runtime_error("CompGraph has no layers!");
                }
                
                // Get last layer's point cloud
                std::shared_ptr<PointCloud> pc_ptr = self.layers.back().point_cloud;
                if (!pc_ptr) {
                    throw std::runtime_error("Last layer point_cloud is null!");
                }
                
                PointCloud& pc = *pc_ptr;
                const size_t N = pc.points.size();
                
                // Get buffer info
                py::buffer_info bF = dLdF_render.request();
                py::buffer_info bX = dLdx_render.request();
                
                // Validate shapes
                if (bF.ndim != 3 || bF.shape[1] != 3 || bF.shape[2] != 3) {
                    throw std::runtime_error(
                        "dLdF must be (N,3,3), got (" + 
                        std::to_string(bF.shape[0]) + "," + 
                        std::to_string(bF.shape[1]) + "," + 
                        std::to_string(bF.shape[2]) + ")"
                    );
                }
                
                if (bX.ndim != 2 || bX.shape[1] != 3) {
                    throw std::runtime_error(
                        "dLdx must be (N,3), got (" + 
                        std::to_string(bX.shape[0]) + "," + 
                        std::to_string(bX.shape[1]) + ")"
                    );
                }
                
                if ((size_t)bF.shape[0] != N || (size_t)bX.shape[0] != N) {
                    throw std::runtime_error(
                        "Shape mismatch: point cloud has " + 
                        std::to_string(N) + " points, but got dLdF with " + 
                        std::to_string(bF.shape[0]) + " and dLdx with " + 
                        std::to_string(bX.shape[0])
                    );
                }
                
                // Get data pointers
                const float* gF = static_cast<const float*>(bF.ptr);
                const float* gX = static_cast<const float*>(bX.ptr);
                
                // Accumulate gradients
                #pragma omp parallel for
                for (int i = 0; i < (int)N; ++i) {
                    MaterialPoint& pt = pc.points[i];
                    
                    // Build Mat3 from row-major numpy array
                    Mat3 dF_grad;
                    dF_grad(0,0) = gF[i*9 + 0];
                    dF_grad(0,1) = gF[i*9 + 1];
                    dF_grad(0,2) = gF[i*9 + 2];
                    dF_grad(1,0) = gF[i*9 + 3];
                    dF_grad(1,1) = gF[i*9 + 4];
                    dF_grad(1,2) = gF[i*9 + 5];
                    dF_grad(2,0) = gF[i*9 + 6];
                    dF_grad(2,1) = gF[i*9 + 7];
                    dF_grad(2,2) = gF[i*9 + 8];
                    
                    // Build Vec3
                    Vec3 dx_grad;
                    dx_grad(0) = gX[i*3 + 0];
                    dx_grad(1) = gX[i*3 + 1];
                    dx_grad(2) = gX[i*3 + 2];
                    
                    // Accumulate to existing gradients
                    pt.dLdF += dF_grad;
                    pt.dLdx += dx_grad;
                }
            },
            py::arg("dLdF_render"),
            py::arg("dLdx_render"),
            R"pbdoc(
                Accumulate render loss gradients to the last layer.
                
                Args:
                    dLdF_render: (N,3,3) numpy array of ∂L_render/∂F
                    dLdx_render: (N,3) numpy array of ∂L_render/∂x
            )pbdoc")
        
        // Carry-over function
        .def("promote_last_as_initial",
            [](CompGraph& self, bool carry_grid) {
                if (self.layers.empty()) return;
                const size_t last = self.layers.size() - 1;
                self.layers.front().point_cloud = self.layers[last].point_cloud;
                if (carry_grid) self.layers.front().grid = self.layers[last].grid;
                self.layers.resize(1);
            },
            py::arg("carry_grid") = false,
            "Promote the last layer's state to the front layer")

         // Single timestep optimization
        .def("optimize_single_timestep", 
            [](CompGraph& self, int timestep_idx, int max_gd_iters, int current_episode, 
               float initial_alpha, int max_line_search_iters) {
                self.OptimizeSingleTimestep(timestep_idx, max_gd_iters, current_episode, 
                                           initial_alpha, max_line_search_iters);
            },
            py::arg("timestep_idx"),
            py::arg("max_gd_iters") = 1,
            py::arg("current_episode") = 0,
            py::arg("initial_alpha") = 1.0f,
            py::arg("max_line_search_iters") = 10,
            R"pbdoc(
                Optimize a single timestep with accumulated gradients.
                
                Args:
                    timestep_idx: Which timestep to optimize (0 to num_layers-2)
                    max_gd_iters: Number of gradient descent iterations (default: 1)
                    current_episode: Current episode number (default: 0)
                    initial_alpha: Initial step size for line search (default: 1.0)
                    max_line_search_iters: Maximum line search iterations (default: 10)
                
                This is useful for interleaved E2E training where you want to
                optimize one timestep at a time while injecting render gradients.
            )pbdoc")
        
        // Get point cloud at specific timestep
        .def("get_point_cloud_at_timestep",
            [](const CompGraph& self, int timestep_idx) -> std::shared_ptr<PointCloud> {
                if (timestep_idx < 0 || timestep_idx >= (int)self.layers.size()) {
                    throw std::out_of_range(
                        "Timestep index " + std::to_string(timestep_idx) + 
                        " out of range [0, " + std::to_string(self.layers.size()) + ")"
                    );
                }
                return self.layers[timestep_idx].point_cloud;
            },
            py::arg("timestep_idx"),
            py::return_value_policy::reference,
            R"pbdoc(
                Get the point cloud at a specific timestep.
                
                Args:
                    timestep_idx: Which timestep to access (0 to num_layers-1)
                
                Returns:
                    PointCloud object at that timestep
                
                Note: This is similar to get_point_cloud() but with clearer naming
                for timestep-based access.
            )pbdoc")

        .def("set_up_comp_graph", &CompGraph::SetUpCompGraph,
            py::arg("num_layers"),
            "Setup computation graph with specified number of layers")
        
        .def("compute_forward_pass", 
            static_cast<void (CompGraph::*)(size_t, int)>(&CompGraph::ComputeForwardPass),
            py::arg("start_layer"),
            py::arg("current_episode"),
            "Run forward simulation from start_layer to end")
        
        .def("compute_backward_pass",
            static_cast<void (CompGraph::*)(size_t)>(&CompGraph::ComputeBackwardPass),
            py::arg("control_layer"),
            "Run backward propagation from end to control_layer")
        .def("set_render_gradients",
            [](CompGraph& self,
                py::array_t<float> dLdF_render,
                py::array_t<float> dLdx_render
            ) {
                // Validate inputs
                if (self.layers.empty()) {
                    throw std::runtime_error("CompGraph has no layers!");
                }
                
                py::buffer_info bF = dLdF_render.request();
                py::buffer_info bX = dLdx_render.request();
                
                // Validate shapes
                if (bF.ndim != 3 || bF.shape[1] != 3 || bF.shape[2] != 3) {
                    throw std::runtime_error(
                        "dLdF must be (N,3,3), got shape (" + 
                        std::to_string(bF.shape[0]) + "," + 
                        std::to_string(bF.shape[1]) + "," + 
                        std::to_string(bF.shape[2]) + ")"
                    );
                }
                
                if (bX.ndim != 2 || bX.shape[1] != 3) {
                    throw std::runtime_error(
                        "dLdx must be (N,3), got shape (" + 
                        std::to_string(bX.shape[0]) + "," + 
                        std::to_string(bX.shape[1]) + ")"
                    );
                }
                
                const size_t N = bF.shape[0];
                if ((size_t)bX.shape[0] != N) {
                    throw std::runtime_error(
                        "Shape mismatch: dLdF has " + std::to_string(N) + 
                        " points, but dLdx has " + std::to_string(bX.shape[0])
                    );
                }
                
                // Store render gradients in CompGraph
                self.stored_render_grad_F_.resize(N * 9);
                self.stored_render_grad_x_.resize(N * 3);
                
                const float* gF = static_cast<const float*>(bF.ptr);
                const float* gX = static_cast<const float*>(bX.ptr);
                
                std::memcpy(self.stored_render_grad_F_.data(), gF, N * 9 * sizeof(float));
                std::memcpy(self.stored_render_grad_x_.data(), gX, N * 3 * sizeof(float));
                
                self.has_render_grads_ = true;
                self.render_grad_num_points_ = N;
                
                std::cout << "[C++] Stored render gradients for " << N << " points" << std::endl;
                
                // Optional: Print gradient norms for debugging
                float norm_F = 0.0f, norm_x = 0.0f;
                for (size_t i = 0; i < N * 9; ++i) {
                    norm_F += gF[i] * gF[i];
                }
                for (size_t i = 0; i < N * 3; ++i) {
                    norm_x += gX[i] * gX[i];
                }
                norm_F = std::sqrt(norm_F);
                norm_x = std::sqrt(norm_x);
                
                std::cout << "[C++]   ||dL_render/dF|| = " << norm_F << std::endl;
                std::cout << "[C++]   ||dL_render/dx|| = " << norm_x << std::endl;
            },
            py::arg("dLdF_render"),
            py::arg("dLdx_render"),
            R"pbdoc(
                Set render loss gradients for the next optimization pass.
                
                These gradients will be automatically injected into the last layer
                during the next compute_backward_pass() call, enabling joint
                optimization of L_tot = L_phys + L_render.
                
                Args:
                    dLdF_render: (N,3,3) numpy array of ∂L_render/∂F
                    dLdx_render: (N,3) numpy array of ∂L_render/∂x
                
                Example:
                    >>> # After computing render loss
                    >>> loss_render.backward()
                    >>> dLdF = F.grad.cpu().numpy()
                    >>> dLdx = x.grad.cpu().numpy()
                    >>> cg.set_render_gradients(dLdF, dLdx)
                    >>> 
                    >>> # Next physics optimization will use L_tot
                    >>> cg.compute_forward_pass(0, episode)
                    >>> cg.compute_backward_pass(0)  # ← L_phys + L_render
                    >>> cg.optimize_single_timestep(0)
            )pbdoc")
        
        .def("clear_render_gradients",
            [](CompGraph& self) {
                self.stored_render_grad_F_.clear();
                self.stored_render_grad_x_.clear();
                self.has_render_grads_ = false;
                self.render_grad_num_points_ = 0;
                std::cout << "[C++] Cleared render gradients" << std::endl;
            },
            R"pbdoc(
                Clear stored render gradients.
                
                Call this at the end of each episode or when you want to reset
                the gradient accumulation state.
                
                Example:
                    >>> # After all passes are done
                    >>> cg.clear_render_gradients()
            )pbdoc")
        
        .def("has_render_gradients",
            [](const CompGraph& self) -> bool {
                return self.has_render_grads_;
            },
            R"pbdoc(
                Check if render gradients are currently stored.
                
                Returns:
                    bool: True if render gradients are available, False otherwise
                
                Example:
                    >>> if cg.has_render_gradients():
                    ...     print("Will optimize with L_tot = L_phys + L_render")
                    ... else:
                    ...     print("Will optimize with L_phys only")
            )pbdoc")
        
        .def("get_render_gradient_info",
            [](const CompGraph& self) -> py::dict {
                py::dict info;
                info["has_gradients"] = self.has_render_grads_;
                info["num_points"] = (int)self.render_grad_num_points_;
                
                if (self.has_render_grads_) {
                    // Compute gradient norms
                    float norm_F = 0.0f, norm_x = 0.0f;
                    for (size_t i = 0; i < self.stored_render_grad_F_.size(); ++i) {
                        float val = self.stored_render_grad_F_[i];
                        norm_F += val * val;
                    }
                    for (size_t i = 0; i < self.stored_render_grad_x_.size(); ++i) {
                        float val = self.stored_render_grad_x_[i];
                        norm_x += val * val;
                    }
                    info["grad_F_norm"] = std::sqrt(norm_F);
                    info["grad_x_norm"] = std::sqrt(norm_x);
                } else {
                    info["grad_F_norm"] = 0.0f;
                    info["grad_x_norm"] = 0.0f;
                }
                
                return info;
            },
            R"pbdoc(
                Get information about stored render gradients.
                
                Returns:
                    dict: Dictionary with keys:
                        - 'has_gradients': bool
                        - 'num_points': int
                        - 'grad_F_norm': float (L2 norm of ∂L/∂F)
                        - 'grad_x_norm': float (L2 norm of ∂L/∂x)
                
                Example:
                    >>> info = cg.get_render_gradient_info()
                    >>> print(f"Render grad norm: {info['grad_F_norm']:.6e}")
            )pbdoc");

    // --- 4. Utilities ---
    m.def("load_point_cloud_from_obj", [](const std::string& obj_path, const OptInput& opt) {
        std::shared_ptr<PointCloud> pc;
        float point_dx = opt.grid_dx / (float)opt.points_per_cell_cuberoot;
        bool success = GeometryLoading::LoadMPMPointCloudFromObj(
            obj_path, pc, point_dx, opt.p_density, opt.lam, opt.mu);
        if (!success) throw std::runtime_error("Failed to load PointCloud from: " + obj_path);
        return pc;
    }, "Load PointCloud from OBJ file");

    m.def("calculate_lame_parameters", [](float young_mod, float poisson) {
        float lam, mu;
        CalculateLameParameters(young_mod, poisson, lam, mu);
        return std::make_pair(lam, mu);
    }, "Calculate Lame parameters");

    m.def("p2g", [](std::shared_ptr<PointCloud> pc, std::shared_ptr<Grid> grid) {
        if (!pc || !grid) throw std::runtime_error("PointCloud or Grid is null.");
        SingleThreadMPM::P2G(*pc, *grid, 0.0f, 0.0f);
    }, "Rasterize PointCloud mass to Grid (P2G)");

    m.def("calculate_point_cloud_volumes", [](std::shared_ptr<PointCloud> pc, std::shared_ptr<Grid> grid) {
        if (!pc || !grid) throw std::runtime_error("PointCloud or Grid is null.");
        SingleThreadMPM::CalculatePointCloudVolumes(*pc, *grid);
    }, "Calculate PointCloud volumes");

    m.def("get_positions_from_pc", [](std::shared_ptr<PointCloud> pc) {
        if (!pc) throw std::runtime_error("PointCloud is null.");
        return pc->GetPointPositions();
    }, "Get positions array from PointCloud");
}