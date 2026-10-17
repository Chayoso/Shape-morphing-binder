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

_run_config = SR.run_config


def run_config_zero(args, cfg0, prep):
    cfg = _run_config(args, cfg0, prep)
    object.__setattr__(cfg, "iters", 0)
    return cfg


SR.run_config = run_config_zero

if __name__ == "__main__":
    SR.main()
