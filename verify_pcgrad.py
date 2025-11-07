#!/usr/bin/env python
"""
PCGrad Verification Script
Checks if PCGrad is working correctly in the pipeline
"""

import sys
import re
from pathlib import Path

def verify_pcgrad_from_logs(log_file: str):
    """Verify PCGrad is working from log file."""

    print(f"\n{'='*70}")
    print(f"PCGrad Verification Report")
    print(f"{'='*70}\n")

    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print(f"\nTo generate logs, run:")
        print(f"  python run.py -c config.yaml --png 2>&1 | tee logs/test.log")
        return False

    with open(log_file, 'r') as f:
        content = f.read()

    # Check 1: PCGrad is enabled
    pcgrad_enabled_pattern = r"use_pcgrad.*True|PCGrad.*enabled"
    pcgrad_enabled = bool(re.search(pcgrad_enabled_pattern, content, re.IGNORECASE))

    print(f"✅ Check 1: PCGrad Enabled")
    print(f"   Status: {'✓ YES' if pcgrad_enabled else '✗ NO'}")

    # Check 2: Conflict detection
    conflict_pattern = r"Conflict.*cos.*=\s*([-\d\.]+)"
    conflicts = re.findall(conflict_pattern, content)

    print(f"\n✅ Check 2: Conflict Detection")
    print(f"   Found {len(conflicts)} conflict checks")

    if conflicts:
        cosines = [float(c) for c in conflicts]
        avg_cosine = sum(cosines) / len(cosines)
        num_conflicts = sum(1 for c in cosines if c < -0.1)

        print(f"   Average cosine: {avg_cosine:.3f}")
        print(f"   Conflicts detected (cos < -0.1): {num_conflicts}/{len(conflicts)}")
        print(f"   Range: [{min(cosines):.3f}, {max(cosines):.3f}]")

    # Check 3: PCGrad application
    pcgrad_applied_pattern = r"PCGrad.*applied|pcgrad_applied.*True"
    pcgrad_applications = len(re.findall(pcgrad_applied_pattern, content, re.IGNORECASE))

    print(f"\n✅ Check 3: PCGrad Application")
    print(f"   Times applied: {pcgrad_applications}")

    if pcgrad_applications > 0:
        # Extract projection scales
        scale_pattern = r"projection_scale.*?(\d+\.\d+)"
        scales = re.findall(scale_pattern, content)
        if scales:
            scales = [float(s) for s in scales]
            print(f"   Projection scale range: [{min(scales):.3f}, {max(scales):.3f}]")
            print(f"   Average projection scale: {sum(scales)/len(scales):.3f}")

    # Check 4: Episode success
    success_pattern = r"Success:\s*(✅|True|✓)"
    failures = re.findall(r"Success:\s*(❌|False)", content)
    successes = len(re.findall(success_pattern, content))

    print(f"\n✅ Check 4: Episode Success")
    print(f"   Successful episodes: {successes}")
    print(f"   Failed episodes: {len(failures)}")

    if successes + len(failures) > 0:
        success_rate = successes / (successes + len(failures)) * 100
        print(f"   Success rate: {success_rate:.1f}%")

    # Check 5: Gradient magnitudes
    render_grad_pattern = r"\|\|g_render\|\|\s*=\s*([\d\.e\-+]+)"
    phys_grad_pattern = r"\|\|g_phys\|\|\s*=\s*([\d\.e\-+]+)"

    render_grads = [float(g) for g in re.findall(render_grad_pattern, content)]
    phys_grads = [float(g) for g in re.findall(phys_grad_pattern, content)]

    print(f"\n✅ Check 5: Gradient Magnitudes")
    if render_grads and phys_grads:
        print(f"   ||g_render|| range: [{min(render_grads):.3e}, {max(render_grads):.3e}]")
        print(f"   ||g_phys||   range: [{min(phys_grads):.3e}, {max(phys_grads):.3e}]")

        ratios = [r/p for r, p in zip(render_grads, phys_grads) if p > 1e-12]
        if ratios:
            print(f"   Ratio (render/phys): [{min(ratios):.3e}, {max(ratios):.3e}]")
            avg_ratio = sum(ratios) / len(ratios)
            print(f"   Average ratio: {avg_ratio:.3e}")

            if avg_ratio > 100:
                print(f"   ⚠️  WARNING: Render gradients still much larger than physics!")
            elif avg_ratio < 0.1:
                print(f"   ⚠️  WARNING: Render gradients too small compared to physics!")
            else:
                print(f"   ✓ Gradients are reasonably balanced")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    issues = []
    if not pcgrad_enabled:
        issues.append("PCGrad is not enabled")
    if conflicts and num_conflicts == 0:
        issues.append("No conflicts detected (may be okay if gradients are aligned)")
    if pcgrad_applications == 0 and num_conflicts > 0:
        issues.append("Conflicts detected but PCGrad not applied!")
    if successes + len(failures) > 0 and len(failures) / (successes + len(failures)) > 0.2:
        issues.append(f"High failure rate: {len(failures) / (successes + len(failures)) * 100:.1f}%")

    if issues:
        print(f"\n⚠️  Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ All checks passed! PCGrad is working correctly.")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_pcgrad.py <log_file>")
        print("\nExample:")
        print("  python run.py -c config.yaml --png 2>&1 | tee logs/test.log")
        print("  python verify_pcgrad.py logs/test.log")
        sys.exit(1)

    log_file = sys.argv[1]
    success = verify_pcgrad_from_logs(log_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
