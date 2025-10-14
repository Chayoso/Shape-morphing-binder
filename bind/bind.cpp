#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

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
    m.doc() = "Python bindings for the DiffMPM core engine";

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
            // Return dFc as (N,3,3) NumPy array
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
            // Return F + dFc as (N,3,3) NumPy array
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
        }, "Return total deformation F_total = F + dFc as (N, 3, 3) array")
;

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
        // NEW: carry-over like the GUI version
        .def("promote_last_as_initial",
            [](CompGraph& self, bool carry_grid) {
                if (self.layers.empty()) return;
                const size_t last = self.layers.size() - 1;
                self.layers.front().point_cloud = self.layers[last].point_cloud;
                if (carry_grid) self.layers.front().grid = self.layers[last].grid;
                self.layers.resize(1);
            },
            py::arg("carry_grid") = false,
            "Promote the last layer's state to the front layer and drop the history."
        );

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
    }, "Calculate Lame parameters from Young's modulus and Poisson's ratio");

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
