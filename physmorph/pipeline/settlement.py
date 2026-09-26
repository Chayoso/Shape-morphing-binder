"""Arrival evidence is evaluated on the accepted state, independently of pacing."""
from dataclasses import dataclass

from physmorph.compute import array_api as np


def _require_start_mask(x, images, radius, start):
    if images is None or radius is None or start is None or np.shape(start) != (len(x),):
        raise ValueError("confirmed arrival requires full plan images and a window-start mask")


def accepted_arrivals(x, images, radius, fallback=None, *, require_start=False):
    x = np.asarray(x)
    if require_start:
        _require_start_mask(x, images, radius, fallback)
    if images is None or radius is None:
        return np.zeros(len(x), bool) if fallback is None else np.asarray(fallback, bool).copy()
    images = np.asarray(images)
    if images.shape != x.shape or not np.isfinite(radius) or radius <= 0:
        raise ValueError("arrival images must match positions and radius must be positive")
    arrived = np.isfinite(x).all(1) & np.isfinite(images).all(1) & (np.linalg.norm(x - images, axis=1) <= radius)
    return arrived & np.asarray(fallback, bool) if require_start else arrived


@dataclass(frozen=True)
class PinArrivalEvidence:
    eligible: np.ndarray
    end_fraction: float | None
    evidence: str

    def telemetry(self):
        return {"arrived_end_frac": self.end_fraction,
                "pin_arrival_evidence": self.evidence,
                "pin_arrival_eligible_frac": float(self.eligible.mean())}


def pin_arrival_evidence(x, images, radius, start=None, *, require_start=False):
    """One eligibility mask for pin admission and transit protection.

    Legacy non-paced runs retain their fallback eligibility, with no claimed
    endpoint arrival rate. Confirmation requires both endpoints of the window.
    """
    has_plan = images is not None and radius is not None
    if require_start:
        _require_start_mask(x, images, radius, start)
    if not has_plan and start is None:
        start = np.ones(len(x), bool)
    end = accepted_arrivals(x, images, radius, start)
    eligible = end & np.asarray(start, bool) if require_start else end
    return PinArrivalEvidence(
        eligible=eligible,
        end_fraction=float(end.mean()) if has_plan else None,
        evidence="accepted_full_plan" if has_plan else "legacy_no_arrival_contract",
    )
