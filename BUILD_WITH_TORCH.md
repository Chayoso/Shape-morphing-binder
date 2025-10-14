# Building DiffMPM with PyTorch Support

## 🎯 Overview

Phase 4 adds optional PyTorch C++ API integration to DiffMPM bindings, enabling:
- Direct Torch tensor output from C++ (no NumPy intermediate)
- Gradient flow from DiffMPM → Python → Renderer
- End-to-end differentiable pipeline

## 📋 Prerequisites

### Required
- **Python 3.8+**
- **PyTorch** (with C++ headers)
- **pybind11**
- **CUDA** (optional, for GPU support)

### Install PyTorch

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```bash
python -c "import torch; print(torch.__version__); print(torch.__file__)"
```

## 🔨 Building

### **Default: Auto-Detect PyTorch (Recommended)**

PyTorch가 설치되어 있으면 **자동으로 torch 지원이 활성화**됩니다!

```bash
# 그냥 설치하면 됩니다
pip install -e .
```

**PyTorch 있을 때 (자동 활성화):**
```
[setup.py] ✅ PyTorch integration ENABLED (torch 2.1.0)
[setup.py]    Include paths: ['/path/to/torch/include', ...]
Building extension diffmpm_bindings...
```

**PyTorch 없을 때 (자동 비활성화):**
```
[setup.py] ℹ️  PyTorch not found - building without torch support
[setup.py]    Install PyTorch to enable full gradient flow: pip install torch
```

---

### **Option: PyTorch 지원 명시적으로 제어**

#### Torch 지원 강제 비활성화:
```bash
# PyTorch가 설치되어 있어도 비활성화
export DIFFMPM_WITH_TORCH=0   # Linux/Mac
set DIFFMPM_WITH_TORCH=0       # Windows CMD
$env:DIFFMPM_WITH_TORCH=0      # Windows PowerShell

pip install -e .
```

#### Torch 지원 강제 활성화 (실패 시 에러):
```bash
export DIFFMPM_WITH_TORCH=1   # Linux/Mac
set DIFFMPM_WITH_TORCH=1       # Windows CMD
$env:DIFFMPM_WITH_TORCH=1      # Windows PowerShell

pip install -e .
# PyTorch 없으면 빌드는 되지만 torch 지원은 비활성화됨
```

## ✅ Verification

### Check if torch support is enabled:

```python
import diffmpm_bindings

# Try to access torch methods
pc = diffmpm_bindings.PointCloud(...)

# Check if torch methods exist
if hasattr(pc, 'get_positions_torch'):
    print("✅ PyTorch support enabled!")
    x_torch = pc.get_positions_torch(requires_grad=True)
    print(f"   Tensor shape: {x_torch.shape}")
    print(f"   Requires grad: {x_torch.requires_grad}")
else:
    print("❌ PyTorch support not available (built without DIFFMPM_WITH_TORCH=1)")
    x_numpy = pc.get_positions()
    print(f"   NumPy array shape: {x_numpy.shape}")
```

## 🚀 Usage Examples

### Basic Usage (Torch tensors)

```python
import torch
import diffmpm_bindings

# Load and run simulation
opt = diffmpm_bindings.OptInput()
# ... configure opt ...

input_pc = diffmpm_bindings.load_point_cloud_from_obj("sphere.obj", opt)
target_pc = diffmpm_bindings.load_point_cloud_from_obj("bunny.obj", opt)

cg = diffmpm_bindings.CompGraph(input_pc, input_grid, target_grid)
cg.run_optimization(opt)

# Get results as Torch tensors (Phase 4!)
last = cg.get_num_layers() - 1
pc = cg.get_point_cloud(last)

# ✅ NEW: Direct torch tensor output
x_t = pc.get_positions_torch(requires_grad=True)      # (N, 3) torch.Tensor
F_t = pc.get_def_grads_total_torch(requires_grad=True)  # (N, 3, 3) torch.Tensor

print(f"x_t: {x_t.shape}, device: {x_t.device}, requires_grad: {x_t.requires_grad}")
print(f"F_t: {F_t.shape}, requires_grad: {F_t.requires_grad}")
```

### End-to-End Gradient Flow

```python
import torch
import diffmpm_bindings
from sampling.runtime_surface import synthesize_runtime_surface, default_cfg
from renderer.renderer import GSRenderer3DGS

# 1) DiffMPM simulation
cg = diffmpm_bindings.CompGraph(...)
cg.run_optimization(opt)

pc = cg.get_point_cloud(cg.get_num_layers() - 1)

# 2) Get torch tensors (with gradient)
x_t = pc.get_positions_torch(requires_grad=True)
F_t = pc.get_def_grads_total_torch(requires_grad=True)

# 3) Upsampling (Phase 3: accepts and returns torch)
cfg = default_cfg()
cfg["differentiable"] = True

result = synthesize_runtime_surface(
    x_t, F_t, cfg,
    differentiable=True,
    return_torch=True  # ← Returns torch tensors
)

mu_t = result["points"]  # torch.Tensor with grad
cov_t = result["cov"]    # torch.Tensor

# 4) Rendering (Phase 1: torch support)
renderer = GSRenderer3DGS(...)
out = renderer.render(mu_t, cov_t, return_torch=True)
img_t = out["image"]  # torch.Tensor

# 5) Loss & backprop
target = load_target_image()
loss = torch.nn.functional.mse_loss(img_t, target)

loss.backward()  # ✅ Gradient flows all the way!

# Check gradients
print(f"x_t.grad: {x_t.grad is not None}")    # True (if DiffMPM params learnable)
print(f"F_t.grad: {F_t.grad is not None}")    # True
print(f"mu_t.grad: {mu_t.grad is not None}")  # True
```

### Fallback Mode (NumPy, if torch build fails)

```python
# Works with or without torch support
pc = cg.get_point_cloud(last)

if hasattr(pc, 'get_positions_torch'):
    # Torch mode
    x = pc.get_positions_torch()
else:
    # NumPy fallback
    x = pc.get_positions()
    x = torch.from_numpy(x)  # Convert manually
```

## 🔧 Troubleshooting

### Error: "torch/extension.h: No such file or directory"

**Solution:** PyTorch C++ headers not found.

```bash
# Verify torch installation
python -c "from torch.utils.cpp_extension import include_paths; print(include_paths())"

# Reinstall torch
pip uninstall torch
pip install torch --force-reinstall
```

### Error: "undefined reference to torch::..."

**Solution:** Linking issue. Try:

```bash
# Make sure you're using the same compiler as PyTorch
python -c "import torch; print(torch.__config__.show())"

# Clean build
pip uninstall diffmpm -y
rm -rf build/ *.egg-info/
DIFFMPM_WITH_TORCH=1 pip install -e .
```

### Windows: MSVC version mismatch

**Solution:** Use same MSVC as PyTorch:

```bash
# Check PyTorch MSVC version
python -c "import torch; print(torch.version.cuda)"

# Use matching Visual Studio (e.g., VS2019 for torch 2.0)
# Open "x64 Native Tools Command Prompt for VS 2019"
set DIFFMPM_WITH_TORCH=1
pip install -e .
```

### Build succeeds but get_positions_torch not available

**Cause:** Built without torch support even though DIFFMPM_WITH_TORCH=1

**Check:**
```bash
# Verify environment variable was set
echo $DIFFMPM_WITH_TORCH  # Should show "1"

# Check build output for this line:
# "[setup.py] PyTorch integration enabled (torch X.X.X)"
```

If you see "Building without PyTorch support", the env var wasn't set properly.

## 📊 Performance Comparison

| Mode | Build Time | Runtime | Memory | Gradient |
|------|-----------|---------|--------|----------|
| **NumPy only** | Fast (~30s) | Fast | Low | ❌ |
| **With Torch** | Slow (~2min) | Medium | Medium | ✅ |

**Recommendation:**
- **Development/Inference**: Build without torch (faster, simpler)
- **Training/Research**: Build with torch (enables e2e gradient)

## 🎓 Complete Training Example

See `test_gradient_flow.py` for a complete example:

```bash
python test_gradient_flow.py
```

Expected output:
```
TEST 1: Renderer Gradient Flow
  ✅ Gradient received!
     Gradient norm: 0.123456

TEST 2: Upsampling → Renderer Gradient Flow
  ✅ Gradient received at upsampled points!
     Non-zero gradient: 4523/5000 points (90.5%)

TEST 3: Learning Upsampling Parameters
  ✅ Parameter learned! (changed by 0.03421)

🎉 All tests passed! Gradient flow is working!
```

## 📚 Additional Resources

- **Phase 1 (Renderer)**: `renderer/renderer.py` - `return_torch` parameter
- **Phase 3 (Upsampling)**: `sampling/runtime_surface.py` - torch tensor I/O
- **Phase 4 (DiffMPM)**: `bind/bind.cpp` - torch C++ API integration
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Quick Start**: `QUICK_START.md`

---

**Status:** ✅ Phase 4 implementation complete!

**To build with torch support:**
```bash
DIFFMPM_WITH_TORCH=1 pip install -e .
```

