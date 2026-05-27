"""`sqpack solve` CLI subcommand."""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from datetime import datetime

from ..known_best import KNOWN_BEST
from ..schema import Result
from ..solver import solve as solve_fn


def register(subparsers):
    p = subparsers.add_parser(
        "solve",
        help="Run the full solver pipeline for n squares.",
        description="Target-based feasibility solver for n unit squares.",
    )
    p.add_argument("n", type=int, help="Number of unit squares")
    p.add_argument("--steps", type=int, default=1000,
                   help="Max compression steps per trial (default: 1000)")
    p.add_argument("--compress-restarts", type=int, default=1,
                   help="Compression restarts per drop (default: 1)")
    p.add_argument("--rotations", type=int, default=2,
                   help="Discrete rotations per trial (default: 2)")
    p.add_argument("--numrotate", type=str, default=None,
                   help='Required when --rotations > 1. Exact count per '
                        'non-zero rotation (e.g. "5" or "1,4")')
    p.add_argument("--angle-power", type=int, default=8,
                   help="Angle sampling bias power (default: 8, 1 for uniform)")
    p.add_argument("--s-start", type=float, default=None,
                   help="Container size (default: ceil(sqrt(n)))")
    p.add_argument("--cohort-size", type=int, default=10,
                   help="Feasibles to collect per cohort before refining the best one")
    p.add_argument("--time", type=int, default=None,
                   help="Time budget in seconds")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="Parallel workers (0 = auto-detect)")
    p.add_argument("--refine-slide-rounds", type=int, default=50)
    p.add_argument("--refine-slide-sa-steps", type=int, default=500000)
    p.add_argument("--refine-sa-t-start", type=float, default=0.5)
    p.add_argument("--refine-sa-t-end", type=float, default=1e-7)
    p.add_argument("--refine-bisect-steps", type=int, default=6)
    p.add_argument("--refine-bisect-slide-steps", type=int, default=2000)
    p.add_argument("--refine-bisect-slide-rounds", type=int, default=20)
    p.add_argument("--refine-explode-factor", type=float, default=1.05)
    p.add_argument("--no-polish", action="store_true",
                   help="Disable the theta polish step")
    p.add_argument("--polish-theta-range", type=float, default=1.0)
    p.add_argument("--polish-theta-points", type=int, default=11)
    p.add_argument("--polish-explode", type=float, default=1.05)
    p.add_argument("--polish-sa-rounds", type=int, default=20)
    p.add_argument("--polish-sa-steps", type=int, default=200000)
    p.add_argument("--polish-convergence-tol", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", type=str, default=None,
                   help="Save PNG of the final packing to this path")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for JSON result files (default: output)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run)


def run(args):
    workers = args.workers
    if workers is None:
        workers = 1
    elif workers == 0:
        workers = multiprocessing.cpu_count()

    if args.rotations > 1 and args.numrotate is None:
        raise SystemExit("--numrotate is required when --rotations > 1")
    numrotate = None
    if args.numrotate is not None:
        numrotate = [int(x) for x in args.numrotate.split(",")]
        if len(numrotate) != args.rotations - 1:
            raise SystemExit(
                f"--numrotate needs {args.rotations - 1} values")
        if sum(numrotate) > args.n:
            raise SystemExit("--numrotate sum exceeds n")

    s_start = args.s_start
    if s_start is None:
        s_start = float(math.ceil(math.sqrt(args.n)))

    if args.verbose:
        print(f"sqpack solver: n={args.n}")
        print(f"  s_start={s_start:.4f}, cohort_size={args.cohort_size}")
        print(f"  Steps/trial: {args.steps}, "
              f"Compress restarts: {args.compress_restarts}")
        nr_str = f", numrotate={numrotate}" if numrotate else ""
        print(f"  Rotations: {args.rotations}{nr_str}, Workers: {workers}")
        print(f"  Compression: {args.rotations} planes x 4 directions "
              f"= {args.rotations * 4} slide directions")
        if args.time:
            print(f"  Time budget: {args.time}s")
        if args.n in KNOWN_BEST:
            print(f"  Known optimal: {KNOWN_BEST[args.n]:.10f}")
        print()

    os.makedirs(args.output_dir, exist_ok=True)
    numrotate_str = "_".join(str(x) for x in numrotate) if numrotate else "0"
    json_path = os.path.join(
        args.output_dir,
        f"sqpack_n_{args.n}_rot_{args.rotations}_numrot_{numrotate_str}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    run_args = {
        "n": args.n,
        "steps": args.steps,
        "compress_restarts": args.compress_restarts,
        "rotations": args.rotations,
        "numrotate": numrotate,
        "s_start": s_start,
        "cohort_size": args.cohort_size,
        "angle_power": args.angle_power,
        "time": args.time,
        "workers": workers,
        "seed": args.seed,
    }
    t_run_start = time.time()
    json_top10 = []

    def on_new_best(s, xs, ys, thetas, n_total):
        from ..io import save_result
        squares = [{"x": xs[i], "y": ys[i], "theta": thetas[i]}
                   for i in range(args.n)]
        threshold = json_top10[-1][0] if len(json_top10) >= 10 else float("inf")
        if s < threshold:
            json_top10.append((s, squares))
            json_top10.sort(key=lambda x: x[0])
            if len(json_top10) > 10:
                json_top10.pop()
        elapsed = time.time() - t_run_start
        result = Result(
            n=args.n, s=s, xs=list(xs), ys=list(ys), thetas=list(thetas),
            s_start=s_start, n_total=n_total, n_feasible=0, n_nontrivial=0,
            elapsed=elapsed,
            metadata={"args": run_args, "seed": args.seed},
        )
        save_result(
            result,
            json_path,
            top_10=[{"s": ts, "squares": tsq} for ts, tsq in json_top10],
        )

    if args.verbose:
        print(f"  Output: {json_path}")

    result = solve_fn(
        n=args.n,
        n_compress_steps=args.steps,
        compress_restarts=args.compress_restarts,
        n_rotations=args.rotations,
        numrotate=numrotate,
        angle_power=args.angle_power,
        enable_polish=not args.no_polish,
        s_start=s_start,
        cohort_size=args.cohort_size,
        seed=args.seed,
        verbose=args.verbose,
        time_limit=args.time,
        workers=workers,
        on_new_best=on_new_best,
        refine_slide_rounds=args.refine_slide_rounds,
        refine_slide_sa_steps=args.refine_slide_sa_steps,
        refine_bisect_steps=args.refine_bisect_steps,
        refine_bisect_sa_steps=args.refine_bisect_slide_steps,
        refine_bisect_slide_rounds=args.refine_bisect_slide_rounds,
        refine_explode_factor=args.refine_explode_factor,
        refine_sa_t_start=args.refine_sa_t_start,
        refine_sa_t_end=args.refine_sa_t_end,
        polish_theta_range=args.polish_theta_range,
        polish_theta_points=args.polish_theta_points,
        polish_explode=args.polish_explode,
        polish_sa_rounds=args.polish_sa_rounds,
        polish_sa_steps=args.polish_sa_steps,
        polish_convergence_tol=args.polish_convergence_tol,
    )

    print(f"\n{'=' * 50}")
    print(f"Results: n={result.n}")
    print(f"  Best s:         {result.s:.10f}")
    if result.n in KNOWN_BEST:
        known = KNOWN_BEST[result.n]
        gap = 100.0 * (result.s - known) / known
        print(f"  Known optimal:  {known:.10f}")
        print(f"  Gap:            {gap:.2f}%")
    print(f"  Container:      {result.s_start}")
    print(f"  Total trials:   {result.n_total:,}")
    print(f"  Feasible:       {result.n_feasible:,}")
    print(f"  Non-trivial:    {result.n_nontrivial:,}")
    print(f"  Time:           {result.elapsed:.2f}s")

    if args.save:
        from ..visualization import plot_packing
        plot_packing(result.n, result.s, result.xs, result.ys, result.thetas,
                     title_prefix="sqpack", filename=args.save, show=False)

    return 0
