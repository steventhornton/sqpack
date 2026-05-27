"""`sqpack refine` CLI subcommand."""

from __future__ import annotations

from ..known_best import KNOWN_BEST
from ..refinement import optimize


def register(subparsers):
    p = subparsers.add_parser(
        "refine",
        help="Refine an existing packing JSON in place.",
        description="Apply sweep + coord-descent + SA + global-shrink to a saved packing.",
    )
    p.add_argument("json_file", help="Input JSON result file")
    p.add_argument("--rounds", type=int, default=100,
                   help="Optimization rounds (default: 100)")
    p.add_argument("--sa-steps", type=int, default=100000,
                   help="SA steps per round (default: 100000)")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSON path (default: overwrite input)")
    p.add_argument("--save", default=None,
                   help="Save PNG visualization to this path")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run)


def run(args):
    from ..io import load_payload, save_result

    payload = load_payload(args.json_file)
    n = payload["n"]
    s_orig = payload["s"]
    xs = [sq["x"] for sq in payload["squares"]]
    ys = [sq["y"] for sq in payload["squares"]]
    thetas = [sq["theta"] for sq in payload["squares"]]

    best_s, best_xs, best_ys, best_thetas, elapsed = optimize(
        n, s_orig, xs, ys, thetas,
        rounds=args.rounds, sa_steps=args.sa_steps, verbose=args.verbose,
    )

    print(f"\n{'=' * 50}")
    print(f"Refinement complete: n={n}")
    print(f"  Original s:    {s_orig:.10f}")
    print(f"  Best s:        {best_s:.10f}")
    print(f"  Improvement:   {s_orig - best_s:.10f}")
    if n in KNOWN_BEST:
        gap = 100.0 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
        print(f"  Known optimal: {KNOWN_BEST[n]:.10f}")
        print(f"  Gap:           {gap:.4f}%")
    print(f"  Time:          {elapsed:.2f}s")

    payload["s"] = float(best_s)
    payload["squares"] = [
        {"x": float(best_xs[i]),
         "y": float(best_ys[i]),
         "theta": float(best_thetas[i])}
        for i in range(n)
    ]
    if n in KNOWN_BEST:
        payload["metadata"]["gap_pct"] = (
            100.0 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
        )
    payload["metadata"]["elapsed_seconds"] = (
        float(payload["metadata"].get("elapsed_seconds", 0.0)) + float(elapsed)
    )
    payload.setdefault("history", []).append({
        "stage": "refine",
        "from_s": float(s_orig),
        "to_s": float(best_s),
        "elapsed_seconds": float(elapsed),
    })
    out_path = args.output or args.json_file
    save_result(payload, out_path)
    print(f"  Saved: {out_path}")

    if args.save:
        from ..visualization import plot_packing
        plot_packing(n, best_s, best_xs, best_ys, best_thetas,
                     title_prefix="sqpack refine",
                     filename=args.save, show=False)
    return 0
