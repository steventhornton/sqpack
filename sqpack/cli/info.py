"""`sqpack info` CLI subcommand."""

from __future__ import annotations

from ..io import load_payload
from ..known_best import KNOWN_BEST


def register(subparsers):
    p = subparsers.add_parser(
        "info",
        help="Print a one-screen summary of a result JSON.",
        description="Show n, s, gap, history, and top_10 of a saved packing.",
    )
    p.add_argument("json_file", help="Path to JSON result file")
    p.set_defaults(func=run)


def run(args):
    payload = load_payload(args.json_file)
    n = payload["n"]
    s = payload["s"]
    meta = payload.get("metadata", {})

    print(f"file:           {args.json_file}")
    print(f"schema_version: {payload.get('schema_version')}")
    print(f"n:              {n}")
    print(f"s:              {s:.10f}")
    known = meta.get("known_best") if meta.get("known_best") is not None else KNOWN_BEST.get(n)
    if known is not None:
        gap_pct = meta.get("gap_pct")
        if gap_pct is None:
            gap_pct = 100.0 * (s - known) / known
        print(f"known_best:     {known:.10f}")
        print(f"gap_pct:        {gap_pct:.6f}%")
    if meta:
        print(f"solver:         {meta.get('solver')}")
        print(f"solver_version: {meta.get('solver_version')}")
        print(f"timestamp:      {meta.get('timestamp')}")
        elapsed = meta.get("elapsed_seconds")
        if elapsed is not None:
            print(f"elapsed:        {elapsed:.2f}s")
        n_total = meta.get("n_total_trials")
        if n_total is not None:
            print(f"n_total_trials: {n_total:,}")

    history = payload.get("history", [])
    if history:
        print("history:")
        for h in history:
            from_s = h.get("from_s")
            to_s = h.get("to_s")
            elapsed = h.get("elapsed_seconds")
            from_str = f"{from_s:.10f}" if isinstance(from_s, (int, float)) else "None"
            to_str = f"{to_s:.10f}" if isinstance(to_s, (int, float)) else "None"
            elap_str = (f"{elapsed:.2f}s"
                        if isinstance(elapsed, (int, float)) else "n/a")
            print(f"  {h.get('stage'):<8} {from_str} -> {to_str}  ({elap_str})")

    top10 = payload.get("top_10", [])
    if top10:
        print(f"top_10: {len(top10)} entries")
        for i, entry in enumerate(top10, 1):
            print(f"  {i:2d}. s={entry['s']:.10f}")
    return 0
