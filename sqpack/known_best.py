"""Best-known s(n) values for packing n unit squares in a square.

Data lives in ``sqpack/data/known_best.json``; this module is a thin loader
that exposes the dict as ``KNOWN_BEST`` and the citation as ``SOURCE``.
"""

from __future__ import annotations

import json
from importlib import resources


def _load():
    with resources.files("sqpack.data").joinpath("known_best.json").open("r") as f:
        payload = json.load(f)
    values = {int(k): float(v) for k, v in payload["values"].items()}
    return values, payload.get("source", ""), payload.get("attribution", "")


KNOWN_BEST, SOURCE, ATTRIBUTION = _load()

__all__ = ["KNOWN_BEST", "SOURCE", "ATTRIBUTION"]
