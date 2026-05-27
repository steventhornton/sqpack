"""`sqpack render` CLI subcommand."""

from __future__ import annotations

import os

from ..io import load_payload
from ..visualization import plot_packing


def register(subparsers):
    p = subparsers.add_parser(
        "render",
        help="Render a packing JSON to a PNG.",
        description="Render a square packing JSON file as a PNG.",
    )
    p.add_argument("json_file", help="Path to JSON result file")
    p.add_argument("-o", "--output", default=None,
                   help="Output PNG path (default: same as JSON with .png)")
    p.add_argument("--show", action="store_true",
                   help="Display the plot interactively")
    p.add_argument("--rank", type=int, default=None,
                   help="Render the Nth best from top_10 (1-indexed)")
    p.set_defaults(func=run)


def run(args):
    payload = load_payload(args.json_file)
    n = payload["n"]

    if args.rank is not None:
        top10 = payload.get("top_10", [])
        if not top10:
            print("No top_10 in this JSON file.")
            return 1
        if args.rank < 1 or args.rank > len(top10):
            print(f"Rank {args.rank} out of range (1-{len(top10)})")
            return 1
        entry = top10[args.rank - 1]
        s = entry["s"]
        squares = entry["squares"]
        title = f"sqpack (rank {args.rank})"
    else:
        s = payload["s"]
        squares = payload["squares"]
        title = "sqpack"

    xs = [sq["x"] for sq in squares]
    ys = [sq["y"] for sq in squares]
    thetas = [sq["theta"] for sq in squares]

    if args.output is None:
        base = os.path.splitext(args.json_file)[0]
        suffix = f"_rank{args.rank}" if args.rank else ""
        output = base + suffix + ".png"
    else:
        output = args.output

    plot_packing(n, s, xs, ys, thetas,
                 title_prefix=title,
                 filename=output,
                 show=args.show)
    return 0
