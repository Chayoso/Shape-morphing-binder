# Troubleshooting Common Errors

## Error 1: bob_to_sphere.yaml Failed (Exit Code 1, 1.5s)

### Symptoms
```
2025-11-06 01:13:23 | ERROR    | ❌ Failed: bob_to_sphere.yaml
2025-11-06 01:13:23 | ERROR    |    Return code: 1
2025-11-06 01:13:23 | ERROR    |    Duration: 1.5s
```

### Root Cause
**Typo in config file** - Line 8 of `configs/Chayo/bob_to_sphere.yaml`:

```yaml
target_mesh_path: "assets/shpere.obj"  # ❌ WRONG: "shpere" doesn't exist
```

### Fix Applied
```yaml
target_mesh_path: "assets/isosphere.obj"  # ✅ CORRECT
output_dir: "output/bob/sphere"           # ✅ Also fixed output path
```

### How to Diagnose Similar Issues

When a config fails in 1-2 seconds, it's usually a startup issue:

1. **Check file paths in config:**
   ```bash
   # Verify all paths exist
   python -c "
   import yaml
   with open('configs/Chayo/bob_to_sphere.yaml') as f:
       cfg = yaml.safe_load(f)
   print('Input:', cfg['input_mesh_path'])
   print('Target:', cfg['target_mesh_path'])
   "
   ```

2. **Check if meshes exist:**
   ```bash
   ls assets/*.obj
   ```

3. **Run config directly to see error:**
   ```bash
   python run.py -c configs/Chayo/bob_to_sphere.yaml
   ```

4. **Use capture-output for detailed logs:**
   ```bash
   python run_batch.py --configs configs/Chayo/bob_to_sphere.yaml --capture-output
   ```

---

## Error 2: ModuleNotFoundError: No module named 'diff_gauss'

### Symptoms
```
ModuleNotFoundError: No module named 'diff_gauss'
```

### Root Cause
Missing or incorrectly compiled Gaussian splatting module.

### Fix
```bash
# Navigate to the diff-gaussian-rasterization directory
cd gaussian-splatting/submodules/diff-gaussian-rasterization/

# Rebuild the module
pip install -e .

# Or if that fails, try:
python setup.py develop
```

---

## Common Quick Fixes

### 1. Missing PNG Files
**Issue:** Some episodes don't have render.png

**Cause:** Training ran without `--png` flag

**Fix:**
```bash
# Always use --png flag for visualization
python run.py -c config.yaml --png
python run_batch.py --config-dir configs/Chayo/ --png
```

### 2. File Not Found Errors
**Issue:** Config fails to load mesh files

**Checks:**
```bash
# List available meshes
ls assets/*.obj

# Verify paths in config match actual files
grep "mesh_path" configs/Chayo/*.yaml
```

**Available meshes:**
- `assets/bob.obj`
- `assets/bunny.obj`
- `assets/isosphere.obj` (sphere)
- `assets/spot.obj`

### 3. Output Directory Issues
**Issue:** Permission denied or directory errors

**Fix:**
```bash
# Ensure output directory is writable
chmod -R u+w output/

# Or specify custom output directory
# Edit config: output_dir: "output/my_experiment/"
```

### 4. CUDA Out of Memory
**Issue:** GPU runs out of memory

**Solutions:**
```bash
# Reduce resolution in config
training_resolution_scale: 0.25  # Lower = less memory

# Reduce particle count
points_per_cell_cuberoot: 2  # Lower = less memory

# Don't run in parallel mode
python run_batch.py --config-dir configs/Chayo/  # Remove --parallel
```

---

## Debugging Workflow

### Step 1: Check Config Syntax
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Step 2: Verify File Paths
```bash
# Check all paths in config exist
python -c "
import yaml
from pathlib import Path

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

for key in ['input_mesh_path', 'target_mesh_path']:
    path = Path(cfg[key])
    print(f'{key}: {path} - {'EXISTS' if path.exists() else 'MISSING'}')
"
```

### Step 3: Test Run
```bash
# Short test run to catch errors early
timeout 30 python run.py -c config.yaml
```

### Step 4: Check Logs
```bash
# View latest log
ls -t logs/batch_run_*.log | head -1 | xargs less

# Search for errors
grep -i "error\|fail\|exception" logs/batch_run_*.log
```

---

## Batch Runner Error Tracking

### Enable Detailed Logging
```bash
# Capture subprocess output (detailed but slower)
python run_batch.py --config-dir configs/Chayo/ --capture-output

# Check the log afterward
cat logs/batch_run_*.log | grep -A 10 "ERROR"
```

### Common Exit Codes

- **Exit Code 1:** General error (config issue, file not found, Python exception)
- **Exit Code 137:** Out of memory (killed by system)
- **Exit Code -1:** Interrupted by user (Ctrl+C)

### Quick Check All Configs
```bash
# Validate all YAML files
for cfg in configs/Chayo/*.yaml; do
    echo "Checking $cfg..."
    python -c "import yaml; yaml.safe_load(open('$cfg'))" || echo "FAILED: $cfg"
done
```

---

## Prevention Tips

1. **Always use absolute or relative paths consistently**
   ```yaml
   # Good
   input_mesh_path: "assets/bob.obj"

   # Avoid
   input_mesh_path: "../assets/bob.obj"  # Can break from different directories
   ```

2. **Use --png flag by default**
   ```bash
   python run_batch.py --config-dir configs/Chayo/ --png
   ```

3. **Test configs individually before batch runs**
   ```bash
   # Quick test (30 seconds)
   timeout 30 python run.py -c new_config.yaml
   ```

4. **Keep logs organized**
   ```bash
   # Use descriptive log directories
   python run_batch.py --config-dir configs/Chayo/ --log-dir logs/experiment_sphere_morphing/
   ```

5. **Check available GPU memory before large runs**
   ```bash
   nvidia-smi
   ```
