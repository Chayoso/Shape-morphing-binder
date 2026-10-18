"""zero_scene.py [scene_run.py's arguments] -- B9b forensic: scripts/scene_run.py with no optimiser iterations (the run's
config with iters 0, set after its construction, which asks for at least one): every window commits the zero control's
rollout, the scene's uncontrolled continuation (the grid, gravity, the floor, the deformation damping, the outer layer's
relaxation and the minimum spacing below the grid; u stays zero). What the body does on the floor under gravity without
the controls. Run as scene_run.py (CUDA, hyde06)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import physmorph  # noqa: E402,F401  (before torch)
import scripts.scene_run as SR  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402

_run_config = SR.run_config
_post_init = PipelineConfig.__post_init__


def _post_init_zero(self):
    """The config's checks with iters 0 let through: the loss raster's growth (scene/raster.py) copies the config with
    dataclasses.replace, which runs the checks again (the first fx_dragon0 ended there at 6.33 s)."""
    if self.iters == 0:
        object.__setattr__(self, "iters", 1)
        _post_init(self)
        object.__setattr__(self, "iters", 0)
    else:
        _post_init(self)


PipelineConfig.__post_init__ = _post_init_zero


def run_config_zero(args, cfg0, prep):
    cfg = _run_config(args, cfg0, prep)
    object.__setattr__(cfg, "iters", 0)
    return cfg


SR.run_config = run_config_zero

if __name__ == "__main__":
    SR.main()
