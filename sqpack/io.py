"""JSON read/write for sqpack Result objects.

``save_result(result, path)`` always writes the v1 layout documented in
``docs/output-schema.md``. ``load_result(path)`` reads either v1 or the
pre-v1 flat layout (auto-migrating the latter in memory; the on-disk
file is not modified unless the caller calls ``save_result`` explicitly).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .schema import (
    Result,
    SCHEMA_VERSION,
    SOLVER_ID,
    SchemaError,
    is_v1,
    validate_v1,
)
from .known_best import KNOWN_BEST


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _solver_version() -> str:
    from . import __version__
    return __version__


def result_to_v1(result: Result,
                 solver: str | None = None,
                 history: list | None = None,
                 top_10: list | None = None,
                 extra_metadata: dict | None = None) -> dict:
    """Serialize a Result dataclass into the v1 dict layout."""
    known = KNOWN_BEST.get(result.n)
    gap_pct = (100.0 * (result.s - known) / known) if known else None

    metadata = {
        "solver": solver or result.metadata.get("solver") or SOLVER_ID,
        "solver_version": _solver_version(),
        "timestamp": _now_iso(),
        "elapsed_seconds": float(result.elapsed),
        "known_best": known,
        "gap_pct": gap_pct,
        "n_total_trials": int(result.n_total),
        "seed": result.metadata.get("seed"),
        "args": result.metadata.get("args", {}),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "n": int(result.n),
        "s": float(result.s),
        "squares": [
            {"x": float(result.xs[i]),
             "y": float(result.ys[i]),
             "theta": float(result.thetas[i])}
            for i in range(result.n)
        ],
        "metadata": metadata,
        "history": list(history or result.history or []),
    }
    chosen_top_10 = top_10 if top_10 is not None else result.top_10
    if chosen_top_10:
        payload["top_10"] = list(chosen_top_10)
    return payload


def save_result(result: Result | dict,
                path: str,
                *,
                solver: str | None = None,
                history: list | None = None,
                top_10: list | None = None,
                extra_metadata: dict | None = None) -> str:
    """Write a v1 JSON file. Accepts a Result or a pre-built v1 dict.

    Returns the path written. Creates parent directories if needed.
    """
    if isinstance(result, Result):
        payload = result_to_v1(
            result, solver=solver, history=history,
            top_10=top_10, extra_metadata=extra_metadata,
        )
    elif isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        validate_v1(payload)
    else:
        raise TypeError(f"save_result: unsupported type {type(result).__name__}")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _migrate_legacy(payload: dict) -> dict:
    """Map a pre-v1 flat-layout dict into v1 in memory (does not write)."""
    n = payload.get("n")
    s = payload.get("s")
    squares = payload.get("squares")
    if n is None or s is None or squares is None:
        raise SchemaError("legacy payload missing one of: n, s, squares")

    metadata = {
        "solver": SOLVER_ID,
        "solver_version": _solver_version(),
        "timestamp": payload.get("timestamp") or _now_iso(),
        "elapsed_seconds": float(payload.get("elapsed", 0.0)),
        "known_best": payload.get("known_best"),
        "gap_pct": payload.get("gap_pct"),
        "n_total_trials": int(payload.get("n_total", 0)),
        "seed": (payload.get("args") or {}).get("seed"),
        "args": payload.get("args") or {},
    }

    history = []
    if "optimized_from" in payload:
        history.append({
            "stage": "refine",
            "from_s": float(payload["optimized_from"]),
            "to_s": float(s),
            "elapsed_seconds": None,
        })
    if "polished_from" in payload:
        history.append({
            "stage": "polish",
            "from_s": float(payload["polished_from"]),
            "to_s": float(s),
            "elapsed_seconds": None,
        })

    v1 = {
        "schema_version": SCHEMA_VERSION,
        "n": int(n),
        "s": float(s),
        "squares": squares,
        "metadata": metadata,
        "history": history,
    }
    if "top_10" in payload:
        v1["top_10"] = payload["top_10"]
    return v1


def load_result(path: str) -> Result:
    """Load a v1 (or legacy) result JSON and return a Result dataclass.

    Legacy flat-layout files are migrated in memory. Use ``save_result``
    on the returned Result to persist the v1 form.
    """
    with open(path) as f:
        payload = json.load(f)

    if not is_v1(payload):
        payload = _migrate_legacy(payload)
    validate_v1(payload)

    squares = payload["squares"]
    return Result(
        n=int(payload["n"]),
        s=float(payload["s"]),
        xs=[float(sq["x"]) for sq in squares],
        ys=[float(sq["y"]) for sq in squares],
        thetas=[float(sq["theta"]) for sq in squares],
        s_start=float(payload["metadata"].get("args", {}).get("s_start", 0.0) or 0.0),
        n_total=int(payload["metadata"].get("n_total_trials", 0)),
        n_feasible=0,
        n_nontrivial=0,
        elapsed=float(payload["metadata"].get("elapsed_seconds", 0.0)),
        metadata=payload["metadata"],
        history=payload.get("history", []),
        top_10=payload.get("top_10", []),
    )


def load_payload(path: str) -> dict:
    """Lower-level: load and migrate, returning the v1 dict (not a Result)."""
    with open(path) as f:
        payload = json.load(f)
    if not is_v1(payload):
        payload = _migrate_legacy(payload)
    validate_v1(payload)
    return payload
