# Batch Runner with Logging

The `run_batch.py` script runs multiple YAML configs with comprehensive logging.

## Features

- **Timestamped logs**: Every run creates a timestamped log file
- **Console + File output**: Logs to both console (colored) and file (detailed)
- **Summary reports**: Automatic summary generation with statistics
- **Progress tracking**: Real-time progress updates
- **Error capturing**: Detailed error logging with exit codes

## Basic Usage

### Run all configs in a directory
```bash
python run_batch.py --config-dir configs/Chayo/
```

### Run with PNG export enabled
```bash
python run_batch.py --config-dir configs/Chayo/ --png
```

### Run specific configs
```bash
python run_batch.py --configs configs/Chayo/sphere_to_bob.yaml configs/Chayo/sphere_to_spot.yaml --png
```

### Run with custom log directory
```bash
python run_batch.py --config-dir configs/Chayo/ --png --log-dir my_logs/
```

### Run with output capture (logs subprocess output)
```bash
python run_batch.py --config-dir configs/Chayo/ --png --capture-output
```
*Note: This may increase memory usage but provides detailed subprocess logs*

### Run in parallel (experimental)
```bash
python run_batch.py --config-dir configs/Chayo/ --png --parallel --max-workers 2
```
*Warning: Uses more GPU memory!*

## Output Files

After running, you'll find:

### 1. Main log file
`logs/batch_run_20250611_143022.log`

Contains:
- Timestamped events
- Command executions
- Detailed progress
- Error messages
- Complete execution history

Example:
```
2025-06-11 14:30:22 | INFO     | ========================================
2025-06-11 14:30:22 | INFO     | Starting: sphere_to_bob.yaml
2025-06-11 14:30:22 | INFO     | ========================================
2025-06-11 14:30:22 | INFO     | Command: python run.py -c configs/Chayo/sphere_to_bob.yaml --png
2025-06-11 14:35:45 | INFO     | ✅ Completed: sphere_to_bob.yaml
2025-06-11 14:35:45 | INFO     |    Duration: 323.2s (5.4m)
```

### 2. Summary file
`logs/summary_batch_run_20250611_143022.txt`

Contains:
- Success/failure statistics
- Per-config timing
- Exit codes for failures
- Quick overview

Example:
```
================================================================================
BATCH RUN SUMMARY
================================================================================

Timestamp: 2025-06-11 14:45:30
Total configs: 3
Successful:    2 (66.7%)
Failed:        1 (33.3%)
Total time:    645.3s (10.8m)

✅ Successful runs:
  - sphere_to_bob.yaml (323.2s)
  - sphere_to_spot.yaml (280.5s)

❌ Failed runs:
  - sphere_to_bunny.yaml (exit code: 1)

================================================================================
Full log: logs/batch_run_20250611_143022.log
```

## Log Levels

The logger uses different levels:

- **INFO**: Normal execution progress (console + file)
- **WARNING**: Non-critical issues (console + file)
- **ERROR**: Execution failures (console + file, highlighted in red)
- **DEBUG**: Detailed subprocess output (file only, use `--capture-output`)

## Tips

1. **Check logs after failures**: Error messages and stack traces are logged
2. **Use `--capture-output` for debugging**: Captures subprocess stdout/stderr
3. **Keep logs organized**: Use `--log-dir` to specify custom directories
4. **Parallel mode**: Only use if you have multiple GPUs or configs are small

## Example Workflow

```bash
# 1. Run batch with logging
python run_batch.py --config-dir configs/Chayo/ --png --log-dir logs/experiment1/

# 2. Check summary (opens automatically at end, or view manually)
cat logs/experiment1/summary_batch_run_*.txt

# 3. If there are failures, check the full log
less logs/experiment1/batch_run_*.log

# 4. Re-run failed configs individually with more verbose output
python run.py -c configs/Chayo/sphere_to_bunny.yaml --png
```

## Comparison: With vs Without Logging

### Old way (no logging)
```bash
python run.py -c config1.yaml
python run.py -c config2.yaml
python run.py -c config3.yaml
# ❌ No record of what ran
# ❌ Hard to track failures
# ❌ No timing information
```

### New way (with batch runner + logging)
```bash
python run_batch.py --config-dir configs/Chayo/ --png
# ✅ Complete execution log
# ✅ Automatic summary
# ✅ Timing statistics
# ✅ Error tracking
```

## Troubleshooting

**Q: Log files are too large**
A: Don't use `--capture-output` unless debugging. Regular logs are much smaller.

**Q: Where are my PNG files?**
A: Remember to add `--png` flag! PNG files go to `output/<config>/ep{XXX}/` directories.

**Q: Can I run this overnight?**
A: Yes! All progress is logged. Check the summary file the next morning.

**Q: How do I stop a batch run?**
A: Press Ctrl+C. The log will show where it was interrupted.
