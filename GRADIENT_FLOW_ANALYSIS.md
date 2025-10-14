# Gradient Flow Analysis: DiffMPM → Upsampling → Renderer

## 🔴 **현재 상태: 미분이 끊어짐!**

### Pipeline 구조 (run.py)

```python
# Line 232: DiffMPM optimization
cg.run_optimization(opt)  # C++ bindings (✅ 미분 가능한 MPM)

# Line 238-239: ❌ GRADIENT BROKEN HERE
x = _np(pc.get_positions())         # C++ → NumPy 변환
F = _np(pc.get_def_grads_total())    # C++ → NumPy 변환

# Line 242: Upsampling
result = synthesize_runtime_surface(x, F, rs, ...)  # NumPy 입력
mu, cov = result["points"], result["cov"]           # NumPy 출력

# Line 327: Renderer
out = renderer.render(mu_t, cov_t, ...)  # Line 312: @torch.no_grad() ❌
```

---

## 🔍 Gradient Flow 단계별 분석

### 1️⃣ **DiffMPM → Python** ❌ BROKEN

**문제:**
```python
# run.py:238-239
x = _np(pc.get_positions())         # C++ binding → NumPy
F = _np(pc.get_def_grads_total())
```

- `diffmpm_bindings.PointCloud`는 C++ 객체
- `.get_positions()`, `.get_def_grads_total()`은 **NumPy 배열 반환**
- PyTorch autograd graph와 **완전히 단절**

**DiffMPM 내부 (C++):**
```cpp
// DiffMPMLib3D/CompGraph.cpp
// CompGraph는 forward + backward를 C++에서 직접 구현
// Python autograd와 통합 안됨
```

---

### 2️⃣ **Upsampling** ✅ NOW DIFFERENTIABLE (if torch input)

**방금 수정한 부분:**
```python
# sampling/runtime_surface.py
def synthesize_runtime_surface(x_low, F_low, cfg, ..., differentiable=True):
    if differentiable and TORCH_AVAILABLE:
        # Torch operations maintain gradient
        ...
```

✅ **torch tensor 입력 → torch tensor 출력 가능**
❌ **하지만 현재는 NumPy를 받음 → NumPy 반환**

---

### 3️⃣ **Renderer** ❌ EXPLICITLY DISABLED

**문제:**
```python
# renderer/renderer.py:312
@torch.no_grad()  # ❌ Gradient 명시적으로 차단!
def render(self, xyz, cov, ...):
```

**내부 구조:**
```python
# Line 336-337: NumPy → Torch (no gradient)
means3D = _to_torch(xyz.astype(np.float32), device=device)

# Line 353-357: Rasterizer (원래는 미분 가능!)
out = self.rasterizer(
    means3D=means3D, means2D=means2D,
    opacities=opac_t, colors_precomp=colors_t,
    cov3D_precomp=cov_t
)

# Line 387: Torch → NumPy (gradient lost)
rgb_np, depth_np, alpha_np = _parse_rasterizer_outputs(...)
return {"image": rgb_np, ...}  # NumPy 반환
```

---

## 📊 **Gradient Breakpoints 요약**

| 단계 | 현재 상태 | Gradient Flow | 비고 |
|------|-----------|---------------|------|
| **DiffMPM (C++)** | ✅ Differentiable | ❌ **Broken** | C++ → NumPy 변환 |
| **DiffMPM → Python** | ❌ NumPy | ❌ **BROKEN** | `.get_positions()` 반환값 |
| **Upsampling** | ✅ Torch-ready | ⚠️ Unused | NumPy 입력받음 |
| **Upsampling → Renderer** | ❌ NumPy | ❌ Broken | NumPy 전달 |
| **Renderer** | ❌ `@no_grad()` | ❌ **BLOCKED** | 명시적 차단 |
| **Rasterizer** | ✅ Differentiable | ❌ Unused | 원래 가능하지만 래퍼가 막음 |

---

## ✅ **해결 방법**

### **Option 1: End-to-End Training (Full Gradient)**

DiffMPM부터 Renderer까지 모두 연결:

1. **DiffMPM bindings 수정** (C++ → Torch bridge)
   ```python
   # bind/bind.cpp에 추가
   torch::Tensor get_positions_torch();  # NumPy 대신 Torch
   torch::Tensor get_def_grads_torch();
   ```

2. **run.py 수정**
   ```python
   # Torch mode로 전환
   x_t = pc.get_positions_torch().requires_grad_(True)
   F_t = pc.get_def_grads_torch().requires_grad_(True)
   
   result = synthesize_runtime_surface(x_t, F_t, rs, differentiable=True)
   mu_t = result["points"]  # Torch tensor with grad
   
   # Renderer도 gradient 유지
   img_t = renderer.render_torch(mu_t, cov_t)  # 새로운 메서드
   
   loss = compute_loss(img_t, target)
   loss.backward()  # 전체 파이프라인 backprop!
   ```

3. **renderer.py 수정**
   ```python
   # @torch.no_grad() 제거!
   def render_torch(self, xyz_t, cov_t, ...):
       """Differentiable version that keeps gradient."""
       # 내부적으로 torch tensor 유지
       out = self.rasterizer(...)
       return out  # Torch tensor 반환
   ```

---

### **Option 2: Partial Training (일부만)**

#### **2A: Renderer만 미분 가능**
```python
# DiffMPM은 고정, Renderer만 학습
x, F = fixed_from_mpm()  # NumPy, no grad
mu, cov = synthesize_runtime_surface(x, F)  # NumPy

# Torch로 변환하고 학습
mu_t = torch.from_numpy(mu).requires_grad_(True)
cov_t = torch.from_numpy(cov).requires_grad_(True)

img_t = renderer.render_torch(mu_t, cov_t)
loss.backward()  # mu, cov에 대한 gradient만
```

#### **2B: Upsampling 파라미터만 학습**
```python
# Upsampling 내부 파라미터를 학습 가능하게
class LearnableUpsampler(nn.Module):
    def __init__(self):
        self.sigma0 = nn.Parameter(torch.tensor(0.02))
        self.thickness = nn.Parameter(torch.tensor(0.12))
    
    def forward(self, x, F):
        cfg = {"sigma0": self.sigma0, ...}
        return synthesize_runtime_surface(x, F, cfg, differentiable=True)
```

---

### **Option 3: 현재 구조 유지 (Inference Only)**

DiffMPM 자체의 최적화만 사용 (backprop 없이):
```python
# 현재 구조 그대로
# DiffMPM이 C++에서 자체 gradient descent 수행
cg.run_optimization(opt)  # 내부에서 loss 최소화

# Upsampling & Rendering은 visualization만
mu, cov = synthesize_runtime_surface(...)
img = renderer.render(...)  # no grad OK
```

**장점:**
- 코드 수정 최소
- DiffMPM이 이미 최적화 수행

**단점:**
- Rendering loss로 학습 불가
- End-to-end 안됨

---

## 🎯 **권장 사항**

### **목표에 따른 선택:**

1. **단순 시각화/추론**
   - ✅ 현재 구조 그대로 OK
   - DiffMPM의 자체 최적화 활용

2. **Rendering loss로 학습하고 싶다면**
   - ➡️ **Option 1 필요**
   - DiffMPM → Torch bridge 구현
   - Renderer의 `@no_grad()` 제거
   - 전체 파이프라인을 Torch로 연결

3. **Upsampling 파라미터 튜닝**
   - ➡️ **Option 2B**
   - DiffMPM은 고정, upsampling 파라미터만 학습
   - 비교적 간단한 수정

---

## 📝 **결론**

**Q: 지금 DiffMPM - Upsampling - Renderer까지 미분이 이어지나?**

**A: ❌ 아니요, 3곳에서 끊어집니다:**

1. **DiffMPM → Python**: C++ binding이 NumPy 반환
2. **Upsampling → Renderer**: NumPy 전달
3. **Renderer 자체**: `@torch.no_grad()` 데코레이터

**현재는 "inference/visualization only" 구조입니다.**

End-to-end training을 원하면 위의 수정이 필요합니다! 🔧

