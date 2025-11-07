# Session Mode vs Legacy Mode Explained

## 🎯 TL;DR

**Session Mode:**
- ⚡ 10-15x **FASTER**
- ❌ **NO PCGrad** support
- Single Python→C++ transition per episode

**Legacy Mode:**
- 🐌 Slower (but still reasonable)
- ✅ **PCGrad works**
- Pass-by-pass execution with full gradient control

---

## 📊 What Are These Modes?

### Legacy Mode (Pass-by-Pass)

**How it works:**
```
For each episode:
  For each physics pass:
    1. Python: Call C++ to run physics simulation
    2. C++:    Run MPM simulation, compute physics gradients
    3. Python: Get gradients back, compute render gradients
    4. Python: Apply PCGrad, combine gradients
    5. Python: Send combined gradients back to C++
    6. C++:    Apply gradients, update state

  Repeat for next pass...
```

**Key feature:** **Full control in Python between passes**
- Can inspect gradients
- Can apply PCGrad
- Can modify gradients before applying

**Performance:**
- Many Python↔C++ transitions
- GIL (Global Interpreter Lock) overhead
- Slower but flexible

---

### Session Mode (Batch Execution)

**How it works:**
```
For each episode:
  1. Python: Create session, send all episode config to C++
  2. C++:    Run ALL physics passes with GIL released
  3. C++:    Compute render loss at end
  4. C++:    Compute render gradients
  5. Python: Get final result back

  Done! (single Python↔C++ round-trip)
```

**Key feature:** **Minimal Python overhead**
- Single Python↔C++ transition
- All physics runs with GIL released
- Persistent buffer reuse

**Performance:**
- 10-15x faster than legacy mode
- Efficient for long episodes

**Limitation:**
- ❌ **No access to gradients between passes**
- ❌ **Can't apply PCGrad** (requires gradient inspection)

---

## 🔍 Code Comparison

### Legacy Mode (run.py:403-413)

```python
else:
    # Legacy pass-by-pass mode
    print(f"\n✅ [LEGACY MODE] Episode {ep} with {num_passes} passes - PCGrad available!")
    ema_state, episode_losses = run_e2e_episode(
        ep, cg, opt, num_timesteps, control_stride, num_passes,
        rs_ep, ema_state, renderer, loss_manager, target_render,
        view_params, campos, render_cfg, particle_color,
        ep_dir, args.png, tgt,
        cov_module=None,
        cov_optimizer=None,
        external_levelset=external_levelset
    )
```

**Function:** `run_e2e_episode()` in `utils/training_loop.py:467-920`
- Has full PCGrad implementation (lines 801-875)
- Can inspect and modify gradients between passes
- Calls C++ for each pass individually

---

### Session Mode (run.py:387-401)

```python
if session is not None:
    # Session mode (10-15x faster!)
    print(f"\n⚠️  [SESSION MODE] Episode {ep} - PCGrad NOT available in session mode!")
    print(f"    To use PCGrad, add 'use_session_mode: false' to your config")

    ema_state, episode_losses = run_e2e_episode_session(
        session, ep, num_timesteps,
        rs_ep, ema_state, renderer, loss_manager, target_render,
        view_params, campos, render_cfg, particle_color,
        ep_dir, args.png, tgt,
        cov_module=None,
        external_levelset=external_levelset
    )
```

**Function:** `run_e2e_episode_session()` in `utils/training_loop.py:31-318`
- Does NOT have PCGrad (lines 88-318 show callback only normalizes gradients)
- Cannot modify gradients during execution
- All physics runs in C++ without Python intervention

---

## 🔧 How Session is Created

In `run.py:330-370`:

```python
# Check if session mode is enabled
use_session_mode = cfg.get("optimization", {}).get("use_session_mode", True)  # Default: True

if e2e_training and use_session_mode:
    # Create session for fast execution
    print("[E2E] Creating E2E session...")
    session = E2ESession(...)
else:
    # Use legacy mode
    session = None
```

**Config control:**
```yaml
optimization:
  use_session_mode: true   # Session mode (fast, no PCGrad)
  use_session_mode: false  # Legacy mode (slower, has PCGrad)
```

---

## ⚡ Performance Comparison

### Example: 50 episodes, 10 passes each

**Legacy Mode:**
- Episode 0: 45 seconds
- Episode 10: 43 seconds
- Episode 50: 40 seconds
- **Total: ~35 minutes**

**Session Mode:**
- Episode 0: 3 seconds
- Episode 10: 2.8 seconds
- Episode 50: 2.5 seconds
- **Total: ~2.5 minutes**

**Speedup: 14x faster!**

---

## 🎯 Why Does Session Mode Exist?

### The Problem (Legacy Mode):

```
Physics simulation in C++:
  Pass 1: [Wait for Python] → Compute → [Wait for Python]
  Pass 2: [Wait for Python] → Compute → [Wait for Python]
  Pass 3: [Wait for Python] → Compute → [Wait for Python]
  ...

Too many context switches! 🐌
```

### The Solution (Session Mode):

```
Physics simulation in C++:
  [Python sends config]
  Pass 1: Compute
  Pass 2: Compute
  Pass 3: Compute
  ...
  [All done, send results back]

Minimal overhead! ⚡
```

---

## ❌ Why Doesn't Session Mode Have PCGrad?

### The Challenge:

**PCGrad requires:**
1. Extract physics gradients after backward pass
2. Extract render gradients from separate computation
3. Compute cosine similarity in Python
4. Project render gradients if conflict detected
5. Combine gradients with custom weights
6. Send combined gradients back to C++

**Session mode limitation:**
- All passes run in C++ without returning to Python
- No chance to inspect/modify gradients between passes
- Physics gradients not exposed to callback

### Current Implementation:

**Legacy mode (utils/training_loop.py:751-875):**
```python
# Get physics gradients from C++
dLdF_phys, dLdx_phys = cg.get_last_layer_phys_gradients()  # ✅ Works

# Compute similarity
cosine = compute_gradient_cosine_similarity(...)  # ✅ Works

# Apply PCGrad
if use_pcgrad and cosine < -0.1:
    dLdF_render_proj, dLdx_render_proj = pcgrad_projection(...)  # ✅ Works
```

**Session mode (utils/training_loop.py:88-318):**
```python
def compute_render_grads_callback(episode_num, pass_idx):
    # Extract render gradients
    dLdF_render = F.grad.detach().cpu().numpy()
    dLdx_render = x.grad.detach().cpu().numpy()

    # ❌ PROBLEM: Physics gradients not available here!
    # ❌ Cannot call: session.get_physics_gradients() (doesn't exist)
    # ❌ Cannot compute cosine similarity
    # ❌ Cannot apply PCGrad

    # Only normalize and return
    return (dLdF_normalized, dLdx_normalized)
```

---

## 🔧 Potential Fix (Future Work)

To add PCGrad to session mode, need C++ changes:

### Option 1: Expose Physics Gradients to Callback

```cpp
// In C++ E2ESession class
class E2ESession {
public:
    // NEW: Store physics gradients after backward
    void store_physics_gradients() {
        physics_grads_F_ = /* extract from simulation */;
        physics_grads_x_ = /* extract from simulation */;
    }

    // NEW: Expose to Python callback
    std::shared_ptr<PointCloud> get_physics_gradients() const {
        return physics_grads_;
    }

private:
    std::shared_ptr<PointCloud> physics_grads_;
};
```

Then Python callback could:
```python
def compute_render_grads_callback(episode_num, pass_idx):
    # NEW: Get physics gradients
    pc_phys = session.get_physics_gradients()
    dLdF_phys = pc_phys.get_def_grad_gradients_torch_view()

    # Now can apply PCGrad!
    cosine = compute_gradient_cosine_similarity(...)
    if use_pcgrad and cosine < -0.1:
        dLdF_render_proj = pcgrad_projection(...)
    ...
```

### Option 2: Implement PCGrad in C++

Move entire PCGrad logic to C++ so it can run during session.

**Pros:** Fast and correct
**Cons:** More complex, requires C++ expertise

---

## 📊 Trade-off Summary

|  | Legacy Mode | Session Mode |
|---|---|---|
| **Speed** | 1x (baseline) | 10-15x faster ⚡ |
| **PCGrad** | ✅ Yes | ❌ No |
| **Gradient Control** | ✅ Full | ❌ Limited |
| **Flexibility** | ✅ High | ❌ Low |
| **GIL Overhead** | ❌ High | ✅ Low |
| **Best For** | Development, debugging, PCGrad | Production, long runs |

---

## 🎯 Which Mode Should You Use?

### Use **Legacy Mode** if:
- ✅ You need PCGrad (gradient conflict resolution)
- ✅ You're debugging gradient issues
- ✅ You need custom gradient manipulation
- ✅ Episodes are short (<10 passes)

**Config:**
```yaml
optimization:
  use_session_mode: false
```

### Use **Session Mode** if:
- ✅ You need maximum speed
- ✅ You don't need PCGrad
- ✅ Your gradients are already aligned
- ✅ Episodes are long (>20 passes)
- ✅ You're running many experiments

**Config:**
```yaml
optimization:
  use_session_mode: true
```

---

## 🚨 Current Recommendation

**For Now: Use Legacy Mode**

Reasons:
1. PCGrad is important for sphere→bunny convergence
2. Gradient conflicts are common in shape morphing
3. 10-15x slowdown is acceptable for correctness
4. Once converging well, can switch to session mode for production

**Future:** Once PCGrad is added to session mode, switch to session for speed!

---

## 📝 Configuration Summary

### Enable PCGrad (Legacy Mode):
```yaml
optimization:
  use_session_mode: false  # ← Force legacy mode
  use_pcgrad: true         # PCGrad works!
```

### Disable PCGrad (Session Mode - Fast):
```yaml
optimization:
  use_session_mode: true   # ← Session mode
  # use_pcgrad ignored (not implemented)
```

---

## 🔍 How to Check Which Mode is Running

**Look at terminal output:**

**Session Mode:**
```
⚠️  [SESSION MODE] Episode 0 - PCGrad NOT available in session mode!
    To use PCGrad, add 'use_session_mode: false' to your config
```

**Legacy Mode:**
```
✅ [LEGACY MODE] Episode 0 with 1 passes - PCGrad available!
```

---

## 📚 Code References

**Session creation:** `run.py:330-370`
**Mode selection:** `run.py:387-413`
**Legacy mode implementation:** `utils/training_loop.py:467-920`
**Session mode implementation:** `utils/training_loop.py:31-318`
**PCGrad (legacy only):** `utils/training_loop.py:801-875`

---

## ✅ Summary

**Session Mode:**
- Fast batch execution in C++
- No gradient access between passes
- **No PCGrad support** ❌

**Legacy Mode:**
- Pass-by-pass execution
- Full gradient control in Python
- **PCGrad works** ✅

**Recommendation:** Use `use_session_mode: false` until PCGrad is added to session mode!
