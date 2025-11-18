"""
run_batch.py - Batch runner for PhysMorph-GS with multiple config files

Runs the shape morphing pipeline with multiple YAML configuration files.
Includes comprehensive logging to both console and file.

Usage:
    # Run all configs in a directory
    python run_batch.py --config-dir configs/Chayo/

    # Run specific config files
    python run_batch.py --configs configs/Chayo/sphere_to_bob.yaml configs/Chayo/sphere_to_spot.yaml

    # Run with PNG export and logging
    python run_batch.py --config-dir configs/Chayo/ --png --log-dir logs/

    # Run with output capture (logs subprocess output)
    python run_batch.py --config-dir configs/Chayo/ --png --capture-output

    # Run in parallel (experimental)
    python run_batch.py --config-dir configs/Chayo/ --parallel --max-workers 2

Output:
    - logs/batch_run_<timestamp>.log - Full execution log
    - logs/summary_batch_run_<timestamp>.txt - Summary report

Author: CHAYO
Version: 2.0 (with comprehensive logging)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, TextIO
import time
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


class LogFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[37m',       # White
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_dir: Path = None, log_level: int = logging.INFO) -> Path:
    """
    Setup logging to both file and console.

    Args:
        log_dir: Directory to store log files (default: ./logs)
        log_level: Logging level

    Returns:
        Path to the log file
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_run_{timestamp}.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler (detailed, no colors)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (colored, less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = LogFormatter(
        '%(levelname)-8s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return log_file


def find_yaml_files(config_dir: str) -> List[Path]:
    """Find all YAML files in the specified directory."""
    config_path = Path(config_dir)
    if not config_path.exists():
        raise ValueError(f"Directory not found: {config_dir}")

    yaml_files = sorted(config_path.glob("*.yaml")) + sorted(config_path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(f"No YAML files found in: {config_dir}")

    return yaml_files


def run_single_config(config_file: Path, extra_args: List[str], verbose: bool = True,
                      capture_output: bool = False) -> tuple:
    """
    Run the program with a single config file.

    Returns:
        tuple: (config_file, return_code, elapsed_time, stdout, stderr)
    """
    logger = logging.getLogger()
    cmd = [sys.executable, "run.py", "-c", str(config_file)] + extra_args

    logger.info("="*80)
    logger.info(f"Starting: {config_file.name}")
    logger.info("="*80)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("")

    start_time = time.time()
    stdout_text = ""
    stderr_text = ""

    try:
        # Run the command
        if capture_output:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout_text = result.stdout
            stderr_text = result.stderr

            # Log captured output
            if stdout_text:
                logger.debug(f"STDOUT:\n{stdout_text}")
            if stderr_text:
                logger.debug(f"STDERR:\n{stderr_text}")
        else:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=None,
                stderr=None
            )

        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info(f"✅ Completed: {config_file.name}")
        logger.info(f"   Duration: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        logger.info("")

        return (config_file, 0, elapsed_time, stdout_text, stderr_text)

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        stdout_text = e.stdout if e.stdout else ""
        stderr_text = e.stderr if e.stderr else ""

        logger.error("")
        logger.error(f"❌ Failed: {config_file.name}")
        logger.error(f"   Return code: {e.returncode}")
        logger.error(f"   Duration: {elapsed_time:.1f}s")

        if stderr_text:
            logger.error(f"   Error output:\n{stderr_text[:1000]}")  # First 1000 chars
        logger.error("")

        return (config_file, e.returncode, elapsed_time, stdout_text, stderr_text)

    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        logger.warning("")
        logger.warning(f"⚠️ Interrupted: {config_file.name}")
        logger.warning(f"   Duration before interrupt: {elapsed_time:.1f}s")
        logger.warning("")
        return (config_file, -1, elapsed_time, "", "")


def run_sequential(config_files: List[Path], extra_args: List[str], verbose: bool = True,
                   capture_output: bool = False) -> List[tuple]:
    """Run configs sequentially."""
    logger = logging.getLogger()
    results = []

    for i, config_file in enumerate(config_files, 1):
        logger.info("")
        logger.info(f"{'#'*80}")
        logger.info(f"Progress: {i}/{len(config_files)} configs")
        logger.info(f"{'#'*80}")
        logger.info("")

        result = run_single_config(config_file, extra_args, verbose, capture_output)
        results.append(result)

        # Stop on first error if requested
        if result[1] != 0:
            logger.warning("")
            logger.warning(f"⚠️ Error encountered in {config_file.name}, stopping batch run")
            logger.warning("")
            break

    return results


def run_parallel(config_files: List[Path], extra_args: List[str], max_workers: int = 2) -> List[tuple]:
    """Run configs in parallel (experimental)."""
    logger = logging.getLogger()
    logger.info("")
    logger.info(f"[Parallel Mode] Running up to {max_workers} configs simultaneously")
    logger.warning("⚠️ Warning: This may consume significant GPU memory!")
    logger.info("")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(run_single_config, config_file, extra_args, verbose=False, capture_output=True): config_file
            for config_file in config_files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            config_file = futures[future]
            try:
                result = future.result()
                results.append(result)

                config, code, elapsed, stdout, stderr = result
                status = "✅" if code == 0 else "❌"
                logger.info(f"{status} {config.name} ({elapsed:.1f}s)")

            except Exception as e:
                logger.error(f"❌ {config_file.name} raised exception: {e}")
                results.append((config_file, -1, 0, "", str(e)))

    return results


def print_summary(results: List[tuple], total_time: float, log_file: Path = None):
    """Print summary of batch run and save to file."""
    logger = logging.getLogger()

    logger.info("")
    logger.info("="*80)
    logger.info("BATCH RUN SUMMARY")
    logger.info("="*80)

    successful = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]

    logger.info("")
    logger.info(f"Total configs: {len(results)}")
    logger.info(f"Successful:    {len(successful)} ({len(successful)/max(1,len(results))*100:.1f}%)")
    logger.info(f"Failed:        {len(failed)} ({len(failed)/max(1,len(results))*100:.1f}%)")
    logger.info(f"Total time:    {total_time:.1f}s ({total_time/60:.1f}m)")

    if successful:
        logger.info("")
        logger.info("✅ Successful runs:")
        for config, code, elapsed, *_ in successful:
            logger.info(f"  - {config.name} ({elapsed:.1f}s, {elapsed/60:.1f}m)")

    if failed:
        logger.info("")
        logger.error("❌ Failed runs:")
        for config, code, elapsed, *_ in failed:
            logger.error(f"  - {config.name} (exit code: {code}, time: {elapsed:.1f}s)")

    logger.info("="*80)

    if log_file:
        logger.info(f"Full log saved to: {log_file}")
        logger.info("="*80)

    # Save summary to separate file
    if log_file:
        summary_file = log_file.parent / f"summary_{log_file.stem}.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("BATCH RUN SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total configs: {len(results)}\n")
                f.write(f"Successful:    {len(successful)} ({len(successful)/max(1,len(results))*100:.1f}%)\n")
                f.write(f"Failed:        {len(failed)} ({len(failed)/max(1,len(results))*100:.1f}%)\n")
                f.write(f"Total time:    {total_time:.1f}s ({total_time/60:.1f}m)\n\n")

                if successful:
                    f.write("\n✅ Successful runs:\n")
                    for config, code, elapsed, *_ in successful:
                        f.write(f"  - {config.name} ({elapsed:.1f}s)\n")

                if failed:
                    f.write("\n❌ Failed runs:\n")
                    for config, code, elapsed, *_ in failed:
                        f.write(f"  - {config.name} (exit code: {code})\n")

                f.write("\n" + "="*80 + "\n")
                f.write(f"Full log: {log_file}\n")

            logger.info(f"Summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary file: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch runner for PhysMorph-GS with comprehensive logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs in directory
  python run_batch.py --config-dir configs/Chayo/

  # Run specific configs
  python run_batch.py --configs sphere_to_bob.yaml sphere_to_spot.yaml

  # Run with PNG export enabled and custom log directory
  python run_batch.py --config-dir configs/Chayo/ --png --log-dir logs/

  # Run with output capture (logs subprocess stdout/stderr)
  python run_batch.py --config-dir configs/Chayo/ --png --capture-output

  # Run in parallel (use with caution!)
  python run_batch.py --config-dir configs/Chayo/ --parallel --max-workers 2

Logging:
  Logs are saved to:
  - logs/batch_run_<timestamp>.log - Full execution log
  - logs/summary_batch_run_<timestamp>.txt - Summary report
        """
    )

    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--config-dir',
        type=str,
        help='Directory containing YAML config files'
    )
    input_group.add_argument(
        '--configs',
        nargs='+',
        type=str,
        help='Specific config files to run'
    )

    # Execution options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run configs in parallel (experimental, may use lots of GPU memory)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='Max parallel workers (default: 2)'
    )

    # Logging options
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to store log files (default: logs)'
    )
    parser.add_argument(
        '--capture-output',
        action='store_true',
        help='Capture and log subprocess output (may increase memory usage)'
    )

    # Arguments to pass to run.py
    parser.add_argument(
        '--png',
        action='store_true',
        help='Export PNG images (passed to run.py)'
    )
    parser.add_argument(
        '--png-dpi',
        type=int,
        default=160,
        help='PNG DPI (default: 160, passed to run.py)'
    )
    parser.add_argument(
        '--png-ptsize',
        type=float,
        default=0.5,
        help='Point size (default: 0.5, passed to run.py)'
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(Path(args.log_dir))
    logger = logging.getLogger()

    logger.info("="*80)
    logger.info("PhysMorph-GS Batch Runner")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Build list of config files
    if args.config_dir:
        config_files = find_yaml_files(args.config_dir)
        logger.info(f"Found {len(config_files)} config files in {args.config_dir}:")
    else:
        config_files = [Path(c) for c in args.configs]
        logger.info(f"Running {len(config_files)} specified config files:")

    for cf in config_files:
        logger.info(f"  - {cf}")

    # Build extra arguments to pass to run.py
    extra_args = []
    if args.png:
        extra_args.append('--png')
    if args.png_dpi != 160:
        extra_args.extend(['--png_dpi', str(args.png_dpi)])
    if args.png_ptsize != 0.5:
        extra_args.extend(['--png_ptsize', str(args.png_ptsize)])

    if extra_args:
        logger.info(f"\nExtra arguments: {' '.join(extra_args)}")

    logger.info(f"Parallel mode: {'enabled' if args.parallel else 'disabled'}")
    if args.parallel:
        logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Capture output: {'enabled' if args.capture_output else 'disabled'}")

    # Confirm before starting
    logger.info("\nPress Enter to start, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        logger.info("\nCancelled by user")
        return 1

    # Run batch
    start_time = time.time()

    try:
        if args.parallel:
            results = run_parallel(config_files, extra_args, args.max_workers)
        else:
            results = run_sequential(config_files, extra_args, verbose=True,
                                   capture_output=args.capture_output)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Batch run interrupted by user")
        results = []

    total_time = time.time() - start_time

    # Print summary
    if results:
        print_summary(results, total_time, log_file)
    else:
        logger.warning("No results to summarize (batch run was interrupted)")

    logger.info("")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # Exit with appropriate code
    failed = [r for r in results if r[1] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
