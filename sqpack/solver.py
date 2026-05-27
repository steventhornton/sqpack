"""Target-based feasibility solver for square packing.

Asks "can n unit squares fit inside a container of side s?" and binary-searches
over s to find the smallest feasible container. Compression uses horizontal,
vertical, and diagonal (along square edge) slides. No rotation perturbation
or reinsert moves.

Public entry points:
    solve(n, ...): run the full pipeline, return a Result.
    Result: dataclass capturing the packing and run stats.

CLI usage:
    sqpack solve N [--rotations R] [--time SECONDS] [-v] [-w WORKERS]
"""

import math
import time

import numpy as np
from numba import njit

from .refinement import refine_with_slides, refine_with_angle_bisection, polish_theta
from .geometry import (
    overlap_y_interval, overlap_x_interval, overlap_depth,
    find_drop_y, slide_diagonal,
    bounding_box_size,
    QUARTER_PI,
)
from .known_best import KNOWN_BEST
from .schema import Result


# ===================================================================
# JIT helpers
# ===================================================================

@njit(cache=True)
def _find_drop_x(y, ci, si, xs, ys, cos_ts, sin_ts, count, exclude):
    """Leftmost resting x at given y (slide from right toward left wall)."""
    h = 0.5 * (abs(ci) + abs(si))
    rest_x = h
    for j in range(count):
        if j == exclude: continue
        x_lo, x_hi, valid = overlap_x_interval(
            y, ci, si, xs[j], ys[j], cos_ts[j], sin_ts[j])
        if valid and x_hi > rest_x:
            rest_x = x_hi
    return rest_x


@njit(cache=True)
def _find_drop_y_from_below(x, ci, si, xs, ys, cos_ts, sin_ts, count,
                            exclude, s):
    """Topmost resting y at given x (slide from bottom toward ceiling)."""
    h = 0.5 * (abs(ci) + abs(si))
    rest_y = s - h
    for j in range(count):
        if j == exclude: continue
        y_lo, y_hi, valid = overlap_y_interval(
            x, ci, si, xs[j], ys[j], cos_ts[j], sin_ts[j])
        if valid and y_lo < rest_y:
            rest_y = y_lo
    return rest_y


@njit(cache=True)
def _find_drop_x_from_left(y, ci, si, xs, ys, cos_ts, sin_ts, count,
                           exclude, s):
    """Rightmost resting x at given y (slide from left toward right wall)."""
    h = 0.5 * (abs(ci) + abs(si))
    rest_x = s - h
    for j in range(count):
        if j == exclude: continue
        x_lo, x_hi, valid = overlap_x_interval(
            y, ci, si, xs[j], ys[j], cos_ts[j], sin_ts[j])
        if valid and x_lo < rest_x:
            rest_x = x_lo
    return rest_x


# ===================================================================
# JIT: Feasibility trial within fixed container [0, s]^2
# ===================================================================

@njit(cache=True)
def _trial_feasibility_jit(n, s, seed, n_compress_steps, compress_restarts,
                           rot_cos, rot_sin, rot_thetas, rot_assignments):
    """Run one feasibility trial in a fixed container [0, s]^2.

    Compression slides squares along axes derived from the rotation set.
    For each step: pick a random square, a random rotational plane, and
    a random direction (one of 4 per plane). Slide to maximum extent
    (contact or wall). Any square can slide along any plane's axes.

    Returns (feasible, n_placed, xs, ys, thetas).
    """
    np.random.seed(seed)

    xs = np.zeros(n)
    ys = np.zeros(n)
    cos_ts = np.zeros(n)
    sin_ts = np.zeros(n)
    thetas = np.zeros(n)

    # --- Phase 1: Drop placement into fixed container ---
    max_drop_attempts = 100
    n_placed = 0
    for i in range(n):
        rot_idx = rot_assignments[i]
        ci = rot_cos[rot_idx]
        si = rot_sin[rot_idx]
        theta = rot_thetas[rot_idx]
        h = 0.5 * (abs(ci) + abs(si))

        lo = h
        hi = s - h
        if hi <= lo:
            return (False, n_placed, xs, ys, thetas)

        placed = False
        for attempt in range(max_drop_attempts):
            if attempt < max_drop_attempts // 2:
                x = lo + (hi - lo) * np.random.random()
                y = find_drop_y(x, ci, si, xs, ys, cos_ts, sin_ts, i, -1)
            else:
                direction = np.random.randint(0, 4)
                if direction == 0:
                    x = lo + (hi - lo) * np.random.random()
                    y = find_drop_y(x, ci, si, xs, ys, cos_ts, sin_ts,
                                     i, -1)
                elif direction == 1:
                    y = lo + (hi - lo) * np.random.random()
                    x = _find_drop_x(y, ci, si, xs, ys, cos_ts, sin_ts,
                                     i, -1)
                elif direction == 2:
                    x = lo + (hi - lo) * np.random.random()
                    y = _find_drop_y_from_below(
                        x, ci, si, xs, ys, cos_ts, sin_ts, i, -1, s)
                else:
                    y = lo + (hi - lo) * np.random.random()
                    x = _find_drop_x_from_left(
                        y, ci, si, xs, ys, cos_ts, sin_ts, i, -1, s)

            if x >= lo and x <= hi and y >= lo and y <= hi:
                xs[i] = x
                ys[i] = y
                cos_ts[i] = ci
                sin_ts[i] = si
                thetas[i] = theta
                n_placed = i + 1
                placed = True
                break
        if not placed:
            return (False, n_placed, xs, ys, thetas)

    # --- Phase 2: Compress within fixed container walls ---
    # Slide along axes derived from the rotation set.
    # For n_rotations planes, there are n_rotations * 4 possible
    # directions. Each step: random square, random plane, random
    # direction, slide to max extent (contact or wall).
    n_rotations = len(rot_cos)

    drop_xs = xs.copy()
    drop_ys = ys.copy()
    drop_cos = cos_ts.copy()
    drop_sin = sin_ts.copy()
    drop_thetas = thetas.copy()

    best_actual_s = 1e30
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_thetas = thetas.copy()

    for _restart in range(compress_restarts):
        for i in range(n):
            xs[i] = drop_xs[i]
            ys[i] = drop_ys[i]
            cos_ts[i] = drop_cos[i]
            sin_ts[i] = drop_sin[i]
            thetas[i] = drop_thetas[i]

        # Random exploration phase
        for step in range(n_compress_steps):
            idx = np.random.randint(0, n)
            k = np.random.randint(0, n_rotations)
            ck = rot_cos[k]
            sk = rot_sin[k]

            d = np.random.randint(0, 4)
            if d == 0:
                dx = ck;  dy = sk
            elif d == 1:
                dx = -ck; dy = -sk
            elif d == 2:
                dx = -sk; dy = ck
            else:
                dx = sk;  dy = -ck

            h_i = 0.5 * (abs(cos_ts[idx]) + abs(sin_ts[idx]))
            t_max = 1e30
            if dx > 1e-15:
                t_max = min(t_max, (s - h_i - xs[idx]) / dx)
            elif dx < -1e-15:
                t_max = min(t_max, (h_i - xs[idx]) / dx)
            if dy > 1e-15:
                t_max = min(t_max, (s - h_i - ys[idx]) / dy)
            elif dy < -1e-15:
                t_max = min(t_max, (h_i - ys[idx]) / dy)

            if t_max <= 1e-12:
                continue

            new_x, new_y = slide_diagonal(
                idx, dx, dy, t_max, xs, ys, cos_ts, sin_ts, n)
            if abs(new_x - xs[idx]) > 1e-12 or abs(new_y - ys[idx]) > 1e-12:
                xs[idx] = new_x
                ys[idx] = new_y

        # Exhaustive rigidity sweep: cycle through every
        # (square, plane, direction) until a full pass has zero movement.
        # Use the ACTUAL rotation angles from placed squares as the
        # sweep planes (not the trial's sampled rot_cos/rot_sin, which
        # may differ by floating-point epsilon from the placed angles).
        sweep_cos = np.empty(n)
        sweep_sin = np.empty(n)
        n_sweep = 0
        for i in range(n):
            ci = cos_ts[i]
            si = sin_ts[i]
            is_dup = False
            for j in range(n_sweep):
                if abs(ci - sweep_cos[j]) < 1e-9 and abs(si - sweep_sin[j]) < 1e-9:
                    is_dup = True
                    break
            if not is_dup:
                sweep_cos[n_sweep] = ci
                sweep_sin[n_sweep] = si
                n_sweep += 1

        max_rigidity_sweeps = 50
        for sweep in range(max_rigidity_sweeps):
            any_moved = False
            for idx in range(n):
                h_i = 0.5 * (abs(cos_ts[idx]) + abs(sin_ts[idx]))
                for k in range(n_sweep):
                    ck = sweep_cos[k]
                    sk = sweep_sin[k]
                    for d in range(4):
                        if d == 0:
                            dx = ck;  dy = sk
                        elif d == 1:
                            dx = -ck; dy = -sk
                        elif d == 2:
                            dx = -sk; dy = ck
                        else:
                            dx = sk;  dy = -ck

                        t_max = 1e30
                        if dx > 1e-15:
                            t_max = min(t_max, (s - h_i - xs[idx]) / dx)
                        elif dx < -1e-15:
                            t_max = min(t_max, (h_i - xs[idx]) / dx)
                        if dy > 1e-15:
                            t_max = min(t_max, (s - h_i - ys[idx]) / dy)
                        elif dy < -1e-15:
                            t_max = min(t_max, (h_i - ys[idx]) / dy)

                        if t_max <= 1e-12:
                            continue

                        new_x, new_y = slide_diagonal(
                            idx, dx, dy, t_max,
                            xs, ys, cos_ts, sin_ts, n)
                        if (abs(new_x - xs[idx]) > 1e-12 or
                                abs(new_y - ys[idx]) > 1e-12):
                            xs[idx] = new_x
                            ys[idx] = new_y
                            any_moved = True
            if not any_moved:
                break

        # Validate no overlaps before saving
        valid_result = True
        for i in range(n):
            for j in range(i + 1, n):
                if overlap_depth(xs[i], ys[i], cos_ts[i], sin_ts[i],
                                      xs[j], ys[j], cos_ts[j],
                                      sin_ts[j]) > 1e-6:
                    valid_result = False
                    break
            if not valid_result:
                break

        if valid_result:
            actual_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
            if actual_s < best_actual_s:
                best_actual_s = actual_s
                for i in range(n):
                    best_xs[i] = xs[i]
                    best_ys[i] = ys[i]
                    best_thetas[i] = thetas[i]

    if best_actual_s > 1e29:
        return (False, n, best_xs, best_ys, best_thetas)
    return (True, n, best_xs, best_ys, best_thetas)


# ===================================================================
# Python wrappers
# ===================================================================

def _build_feasibility_args(n, s_target, n_compress_steps, compress_restarts,
                            n_rotations, numrotate, rng, angle_power=8):
    """Build args for one _trial_feasibility_jit call."""
    rot_thetas = np.empty(n_rotations)
    rot_thetas[0] = 0.0
    for k in range(1, n_rotations):
        rot_thetas[k] = QUARTER_PI * (1.0 - float(rng.random())**angle_power)
    rot_cos = np.cos(rot_thetas)
    rot_sin = np.sin(rot_thetas)

    if numrotate is not None:
        assignments = [0] * (n - sum(numrotate))
        for k, count in enumerate(numrotate):
            assignments.extend([k + 1] * count)
        rot_assignments = np.array(assignments, dtype=np.int64)
        rng.shuffle(rot_assignments)
    else:
        rot_weights = np.ones(n_rotations)
        rot_weights[0] = 2.0
        rot_probs = rot_weights / rot_weights.sum()
        rot_assignments = rng.choice(
            n_rotations, size=n, p=rot_probs).astype(np.int64)

    jit_seed = int(rng.integers(0, 2**31))

    return (n, s_target, jit_seed, n_compress_steps, compress_restarts,
            rot_cos, rot_sin, rot_thetas, rot_assignments)


# ===================================================================
# Validation
# ===================================================================

def _refine_worker(args):
    """Worker for parallel refinement of a single packing."""
    (n, xs, ys, thetas,
     slide_rounds, slide_sa_steps, sa_t_start, sa_t_end,
     explode_factor, bisect_steps, bisect_sa_steps,
     bisect_slide_rounds,
     p_enable_polish,
     p_theta_range, p_theta_points, p_explode,
     p_sa_rounds, p_sa_steps, p_convergence_tol) = args

    # Step 1: SA refinement
    s1, xs_r, ys_r, thetas_r = refine_with_slides(
        n, xs, ys, thetas,
        rounds=slide_rounds, sa_steps=slide_sa_steps,
        sa_t_start=sa_t_start, sa_t_end=sa_t_end)

    # Step 2: Angle bisection
    s2, xs_r2, ys_r2, thetas_r2 = refine_with_angle_bisection(
        n, xs_r, ys_r, thetas_r,
        explode_factor=explode_factor, bisection_steps=bisect_steps,
        n_steps=bisect_sa_steps, slide_rounds=bisect_slide_rounds)

    # Use best so far
    if s2 < s1:
        best_s, best_xs, best_ys, best_thetas = s2, xs_r2, ys_r2, thetas_r2
    else:
        best_s, best_xs, best_ys, best_thetas = s1, xs_r, ys_r, thetas_r

    # Step 3: Polish theta (iterative narrowing) — optional
    if not p_enable_polish:
        return (best_s, best_xs, best_ys, best_thetas)
    s3, xs_p, ys_p, thetas_p = polish_theta(
        n, best_xs, best_ys, best_thetas,
        theta_range=p_theta_range, theta_points=p_theta_points,
        explode_factor=p_explode, sa_rounds=p_sa_rounds,
        sa_steps=p_sa_steps, convergence_tol=p_convergence_tol)
    if s3 < best_s - 1e-14:
        return (s3, xs_p, ys_p, thetas_p)
    return (best_s, best_xs, best_ys, best_thetas)


def _validate(xs, ys, cos_ts, sin_ts, n, tol=1e-8):
    for i in range(n):
        for j in range(i + 1, n):
            d = overlap_depth(xs[i], ys[i], cos_ts[i], sin_ts[i],
                               xs[j], ys[j], cos_ts[j], sin_ts[j])
            if d > tol:
                return False, f"squares {i} and {j} overlap by {d:.2e}"
    return True, None


# ===================================================================
# Solver
# ===================================================================


def _warmup_jit():
    rc = np.array([1.0])
    rs = np.array([0.0])
    rt = np.array([0.0])
    ra = np.array([0], dtype=np.int64)
    _trial_feasibility_jit(1, 3.0, 42, 1, 1, rc, rs, rt, ra)


def _validate_packing(n, xs_v, ys_v, thetas_v):
    """Check a packing has zero overlaps. Returns True if valid."""
    cos_v = [math.cos(t) for t in thetas_v]
    sin_v = [math.sin(t) for t in thetas_v]
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_depth(xs_v[i], ys_v[i], cos_v[i], sin_v[i],
                             xs_v[j], ys_v[j], cos_v[j], sin_v[j]) > 1e-8:
                return False
    return True


def solve(n, n_compress_steps=1000, compress_restarts=1,
          n_rotations=2, numrotate=None, angle_power=8,
          s_start=None, cohort_size=10, seed=None, verbose=False,
          time_limit=None, workers=None, on_new_best=None,
          refine_slide_rounds=50, refine_slide_sa_steps=500000,
          refine_bisect_steps=6, refine_bisect_sa_steps=200000,
          refine_bisect_slide_rounds=20, refine_explode_factor=1.05,
          refine_sa_t_start=0.5, refine_sa_t_end=1e-7,
          enable_polish=True,
          polish_theta_range=1.0, polish_theta_points=11,
          polish_explode=1.05, polish_sa_rounds=20,
          polish_sa_steps=200000, polish_convergence_tol=0.01):
    """Run cohort-based trials at fixed container size s_start until timeout.

    Each cohort collects `cohort_size` feasible packings via repeated
    drop+compress trials. Once a cohort is full, the packing with the
    smallest actual bounding box is refined (SA slides + angle bisection
    + polish). Refined result is fed into a global top-10 tracker.
    Loops forever until `time_limit` is exhausted.
    """
    if workers is None:
        workers = 1
    if s_start is None:
        s_start = float(math.ceil(math.sqrt(n)))

    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 31)
    rng = np.random.default_rng(seed)

    if verbose:
        print("  Compiling JIT...", end=" ", flush=True)
    t_compile = time.time()
    _warmup_jit()
    if verbose:
        print(f"done ({time.time() - t_compile:.1f}s)\n")

    best_s = float('inf')
    best_data = None
    n_total = 0
    n_feasible = 0
    n_nontrivial = 0
    n_cohorts = 0
    t0 = time.time()
    grid_s = float(math.ceil(math.sqrt(n)))

    def time_ok():
        return time_limit is None or (time.time() - t0) < time_limit

    top_k = 10
    top_packings = []  # list of (s, (xs, ys, thetas)) — global top-10

    last_status = t0

    while time_ok():
        # Phase 1: collect `cohort_size` feasibles
        cohort = []  # list of (actual_bbox, xs, ys, thetas)
        cohort_started = time.time()
        while len(cohort) < cohort_size and time_ok():
            trial_args = _build_feasibility_args(
                n, s_start, n_compress_steps, compress_restarts,
                n_rotations, numrotate, rng, angle_power=angle_power)
            feasible, _, xs, ys, thetas = _trial_feasibility_jit(*trial_args)
            n_total += 1

            if feasible:
                n_feasible += 1
                actual = float(bounding_box_size(
                    xs, ys, np.cos(thetas), np.sin(thetas), n))
                if actual < grid_s - 1e-10:
                    n_nontrivial += 1
                cohort.append((actual, xs.tolist(), ys.tolist(),
                               thetas.tolist()))

            # Periodic status during cohort collection
            now = time.time()
            if verbose and now - last_status >= 60.0:
                elapsed = now - t0
                remaining = (time_limit - elapsed) if time_limit else 0
                rate = n_total / elapsed if elapsed > 0 else 0
                best_str = f"{best_s:.8f}" if best_s < 1e10 else "none"
                time_str = f", {remaining:.0f}s left" if time_limit else ""
                print(f"  ... {n_total:,} trials, "
                      f"feasible={n_feasible:,}, "
                      f"nontrivial={n_nontrivial}, "
                      f"cohort={len(cohort)}/{cohort_size}, "
                      f"@ {rate:.0f}/s, best={best_str} "
                      f"[{elapsed:.0f}s{time_str}]", flush=True)
                last_status = now

        if len(cohort) < cohort_size:
            # Time ran out mid-cohort
            break

        n_cohorts += 1

        # Phase 2: pick best of cohort by actual bbox
        cohort.sort(key=lambda x: x[0])
        best_actual, xs_b, ys_b, thetas_b = cohort[0]
        cohort_elapsed = time.time() - cohort_started

        if verbose:
            print(f"  >> Cohort {n_cohorts} full ({cohort_size} feasibles in "
                  f"{cohort_elapsed:.1f}s): best raw s={best_actual:.6f}, "
                  f"refining...", flush=True)

        # Phase 3: refine + polish the cohort best
        refine_task = (n, xs_b, ys_b, thetas_b,
                       refine_slide_rounds, refine_slide_sa_steps,
                       refine_sa_t_start, refine_sa_t_end,
                       refine_explode_factor, refine_bisect_steps,
                       refine_bisect_sa_steps, refine_bisect_slide_rounds,
                       enable_polish,
                       polish_theta_range, polish_theta_points,
                       polish_explode, polish_sa_rounds,
                       polish_sa_steps, polish_convergence_tol)
        s_ref, xs_ref, ys_ref, thetas_ref = _refine_worker(refine_task)

        if not _validate_packing(n, xs_ref, ys_ref, thetas_ref):
            if verbose:
                print(f"    >> Refined packing failed validation, skipping",
                      flush=True)
            continue

        # Phase 4: update global top-10 + best
        threshold = top_packings[-1][0] if len(top_packings) >= top_k \
            else float('inf')
        if s_ref < threshold:
            top_packings.append((s_ref, (xs_ref, ys_ref, thetas_ref)))
            top_packings.sort(key=lambda x: x[0])
            if len(top_packings) > top_k:
                top_packings.pop()

        if verbose:
            gap_str = ""
            if n in KNOWN_BEST:
                gap = 100 * (s_ref - KNOWN_BEST[n]) / KNOWN_BEST[n]
                gap_str = f" (gap: {gap:.4f}%)"
            new_best_str = " [NEW BEST]" if s_ref < best_s - 1e-14 else ""
            print(f"    >> Refined: s={s_ref:.10f}{gap_str}{new_best_str}",
                  flush=True)

        if s_ref < best_s - 1e-14:
            best_s = s_ref
            best_data = (xs_ref, ys_ref, thetas_ref)
            if on_new_best is not None:
                on_new_best(best_s, xs_ref, ys_ref, thetas_ref, n_total)

    elapsed = time.time() - t0

    if best_data is None:
        return Result(n=n, s=float('inf'), xs=[], ys=[], thetas=[],
                         s_start=s_start, n_total=n_total,
                         n_feasible=n_feasible, n_nontrivial=n_nontrivial,
                         elapsed=elapsed)

    xs, ys, thetas = best_data

    cos_ts = [math.cos(t) for t in thetas]
    sin_ts = [math.sin(t) for t in thetas]
    ok, detail = _validate(xs, ys, cos_ts, sin_ts, n)
    if not ok:
        if verbose:
            print(f"  WARNING: validation failed: {detail}")
            print(f"  Discarding invalid result.")
        return Result(n=n, s=float('inf'), xs=[], ys=[], thetas=[],
                         s_start=s_start, n_total=n_total,
                         n_feasible=n_feasible, n_nontrivial=n_nontrivial,
                         elapsed=elapsed)

    return Result(n=n, s=best_s, xs=xs, ys=ys, thetas=thetas,
                     s_start=s_start, n_total=n_total,
                     n_feasible=n_feasible, n_nontrivial=n_nontrivial,
                     elapsed=elapsed)


