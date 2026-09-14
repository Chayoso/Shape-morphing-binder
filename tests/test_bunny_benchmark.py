import copy
import numpy as np
import pytest
from scripts.bunny_response_benchmark import raw_quality, validate_pair_metadata


def pair():
    a = {k: "same" for k in ("fixture", "steps", "camera_radius", "training_views",
          "basis_sha256", "source_sha256", "script_sha256", "raster_sha256")}
    a["optimizer"] = {"objective": "image", "iterations": 12, "fd_strain": .001, "radius": .03}
    b = copy.deepcopy(a); b["optimizer"]["objective"] = "physics"
    return a, b


def test_bunny_comparison_rejects_mislabeled_or_unmatched_arms():
    a, b = pair()
    validate_pair_metadata(a, b)
    for key, value in (("objective", "image"), ("fd_strain", .01), ("iterations", 3)):
        wrong = copy.deepcopy(b); wrong["optimizer"][key] = value
        with pytest.raises(ValueError):
            validate_pair_metadata(a, wrong)
    b["optimizer"]["iterations"] = 24
    validate_pair_metadata(a, b, "strong_physics")
    with pytest.raises(ValueError):
        validate_pair_metadata(a, b)


def test_oblique_clipping_cannot_disappear_from_bunny_quality_gate():
    # Each coordinate is within the box, but an oblique projection leaves it.
    rng = np.random.default_rng(3)
    target = rng.uniform(-.2, .2, (40, 3)).astype(np.float32)
    cloud = target.copy(); cloud[:9] = [ .95, .95, -.95]
    assert (np.abs(cloud) < 1.).all()
    assert raw_quality(cloud, target, 1.)["projected_outside_max"] > 0.
