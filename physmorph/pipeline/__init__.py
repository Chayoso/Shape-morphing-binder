"""PhysMorph v2 blessed path — render feedback drives the physics.

See docs/pipeline_v2.md. One optimisation core (optimizer.optimize_window), one outer
loop (runner.run_pipeline), one config (config.PipelineConfig). The physics-only baseline
is the SAME code path with the render channel off (lambda_auto=0)."""
from .config import PipelineConfig
from .runner import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
