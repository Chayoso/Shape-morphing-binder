"""Resolve traceable material presets into the existing solver's E and nu."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


CATALOG = Path(__file__).with_name("material_presets.json")


def _catalog():
    content = CATALOG.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def material_names() -> tuple[str, ...]:
    return tuple(_catalog()[0]["presets"])


def resolve_material(name: str) -> dict:
    catalog, digest = _catalog()
    raw = catalog["presets"][name]
    mapping = catalog["mapping"]
    young_pa = sum(raw["young"]["range"]) / 2 * mapping["unit_to_pa"][raw["young"]["unit"]]
    nu_data = raw["poisson"]
    nu = nu_data["assumed"] if nu_data["range"] is None else sum(nu_data["range"]) / 2
    if not (math.isfinite(young_pa) and young_pa > 0 and -1 < nu < .5):
        raise ValueError(f"invalid raw E/nu for {name}: {young_pa}, {nu}")
    stress_unit = mapping['young_anchor_pa'] / mapping['young_anchor_effective']
    raw_moduli = {'bulk': young_pa/(3*(1-2*nu)), 'shear': young_pa/(2*(1+nu))}
    effective_moduli = {key: value / stress_unit for key, value in raw_moduli.items()}
    young = young_pa / stress_unit
    return {
        "name": name,
        "catalog_sha256": digest,
        "raw": raw,
        "sources": {key: catalog["sources"][key] for key in {p["source"] for p in raw.values()}},
        "selected": {"young_pa": young_pa, "poisson": nu},
        "mapping": {**mapping, 'stress_unit_pa': stress_unit},
        "derived_moduli": {"raw_pa": raw_moduli, "effective": effective_moduli},
        "effective": {"young": young, "poisson": nu},
    }
