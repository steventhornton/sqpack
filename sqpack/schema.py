"""JSON output schema and Result dataclass.

The Result dataclass is the runtime object the solver returns. ``save_result``
in ``sqpack.io`` serializes it into the v1 JSON layout documented in
``docs/output-schema.md``. ``load_result`` reverses the process and migrates
legacy flat-layout files (no ``schema_version`` key) into v1 on read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

SOLVER_ID = "sqpack.feasibility"


class SchemaError(ValueError):
    """Raised when a JSON payload cannot be parsed as a v1 result."""


@dataclass
class Result:
    """Runtime result of a solve / refine / polish stage.

    Coordinate arrays ``xs``, ``ys``, ``thetas`` are length-``n`` parallel
    lists describing each square's center and rotation. ``s`` is the side
    length of the enclosing axis-aligned square.
    """

    n: int
    s: float
    xs: list
    ys: list
    thetas: list
    s_start: float = 0.0
    n_total: int = 0
    n_feasible: int = 0
    n_nontrivial: int = 0
    elapsed: float = 0.0
    metadata: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    top_10: list = field(default_factory=list)


def validate_v1(payload: dict) -> None:
    """Raise SchemaError if ``payload`` is not a valid v1 result."""
    required_top = {"schema_version", "n", "s", "squares", "metadata"}
    missing = required_top - payload.keys()
    if missing:
        raise SchemaError(f"missing required keys: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {payload['schema_version']!r} != {SCHEMA_VERSION!r}"
        )
    if not isinstance(payload["n"], int):
        raise SchemaError("n must be int")
    if not isinstance(payload["s"], (int, float)):
        raise SchemaError("s must be number")
    squares = payload["squares"]
    if not isinstance(squares, list) or len(squares) != payload["n"]:
        raise SchemaError("squares must be a list of length n")
    for i, sq in enumerate(squares):
        if not isinstance(sq, dict):
            raise SchemaError(f"squares[{i}] must be a dict")
        for k in ("x", "y", "theta"):
            if k not in sq:
                raise SchemaError(f"squares[{i}] missing {k!r}")
    meta = payload["metadata"]
    if not isinstance(meta, dict):
        raise SchemaError("metadata must be a dict")
    for k in ("solver", "timestamp"):
        if k not in meta:
            raise SchemaError(f"metadata.{k} is required")


def is_v1(payload: dict) -> bool:
    return payload.get("schema_version") == SCHEMA_VERSION
