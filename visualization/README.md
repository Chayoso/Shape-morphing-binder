Visualization Utilities
=======================

All visualization-related scripts and helpers now live under this directory to
keep the project root clean. The layout is:

- `scripts/`: Standalone plotting/diagnostic scripts such as
  `visualize_alpha_loss.py`, `visualize_multiscale.py`, etc. Invoke them with
  `python -m visualization.scripts.visualize_alpha_loss ...` or run the files
  directly from this folder.
- `utils.py`: Shared helpers (episode exporters, matplotlib comparisons, etc.)
  that are imported by the training loop and tests.

Any new visualization entry points should be placed here for discoverability.
