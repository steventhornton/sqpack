"""Refinement and polish stages for square packings.

Applies iterative improvement techniques to push an existing packing closer
to optimal:
  1. Multi-axis exhaustive sweep (along actual rotation planes)
  2. Gradient-free coordinate descent (fine-grained per-square)
  3. Simulated annealing with random partial slides
  4. Global shrink: scale all positions inward, then re-resolve via sweeps

Public API:
    refine_with_slides(...), refine_with_angle_bisection(...),
    polish_theta(...), optimize(...)

CLI usage:
    sqpack refine RESULT.json [--rounds N] [--sa-steps N] [-v]
"""

import math
import time

import numpy as np
from numba import njit

from .geometry import (
    overlap_depth, overlap_y_interval, overlap_x_interval,
    overlap_t_interval, slide_diagonal, bounding_box_size,
)
from .known_best import KNOWN_BEST


# ===================================================================
# Numba-compiled optimization routines
# ===================================================================

@njit(cache=True)
def _bbox_constrained_sweep(xs, ys, cos_ts, sin_ts, n,
                            sweep_cos, sweep_sin, n_sweep, max_sweeps):
    """Exhaustive sweep constrained by the current bounding box.

    For each square/plane/direction: slide to max extent but only accept
    the move if it reduces the bounding box. This prevents outward expansion.
    Repeats until no improvement or max_sweeps reached.
    Returns total number of improving moves made.
    """
    total_moves = 0
    best_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

    for sweep in range(max_sweeps):
        any_moved = False
        for idx in range(n):
            for k in range(n_sweep):
                ck = sweep_cos[k]
                sk = sweep_sin[k]
                for d in range(4):
                    if d == 0: dx = ck;  dy = sk
                    elif d == 1: dx = -ck; dy = -sk
                    elif d == 2: dx = -sk; dy = ck
                    else: dx = sk;  dy = -ck

                    new_x, new_y = slide_diagonal(
                        idx, dx, dy, 1e30, xs, ys, cos_ts, sin_ts, n)
                    if (abs(new_x - xs[idx]) < 1e-12 and
                            abs(new_y - ys[idx]) < 1e-12):
                        continue

                    old_x, old_y = xs[idx], ys[idx]
                    xs[idx] = new_x
                    ys[idx] = new_y
                    new_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

                    if new_s < best_s - 1e-14:
                        best_s = new_s
                        any_moved = True
                        total_moves += 1
                    else:
                        xs[idx] = old_x
                        ys[idx] = old_y
        if not any_moved:
            break
    return total_moves


@njit(cache=True)
def _coordinate_descent(xs, ys, cos_ts, sin_ts, n,
                        sweep_cos, sweep_sin, n_sweep,
                        n_steps, step_size):
    """Fine-grained coordinate descent.

    For each square, try small moves in each direction and accept if
    the bounding box shrinks. Step size decreases over iterations.
    """
    best_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
    improved = 0

    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        current_step = step_size * (1.0 - 0.9 * frac)  # anneal from step_size to 0.1*step_size

        for idx in range(n):
            for k in range(n_sweep):
                ck = sweep_cos[k]
                sk = sweep_sin[k]
                for d in range(4):
                    if d == 0: dx = ck;  dy = sk
                    elif d == 1: dx = -ck; dy = -sk
                    elif d == 2: dx = -sk; dy = ck
                    else: dx = sk;  dy = -ck

                    # Try a move of exactly current_step (not max extent)
                    new_x = xs[idx] + dx * current_step
                    new_y = ys[idx] + dy * current_step

                    # Check overlaps
                    has_overlap = False
                    for j in range(n):
                        if j == idx:
                            continue
                        if overlap_depth(new_x, new_y, cos_ts[idx], sin_ts[idx],
                                         xs[j], ys[j], cos_ts[j], sin_ts[j]) > 1e-10:
                            has_overlap = True
                            break
                    if has_overlap:
                        continue

                    # Accept if bounding box improves
                    old_x, old_y = xs[idx], ys[idx]
                    xs[idx] = new_x
                    ys[idx] = new_y
                    new_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
                    if new_s < best_s - 1e-14:
                        best_s = new_s
                        improved += 1
                    else:
                        xs[idx] = old_x
                        ys[idx] = old_y

    return best_s, improved


@njit(cache=True)
def _sa_optimize(xs, ys, cos_ts, sin_ts, n,
                 sweep_cos, sweep_sin, n_sweep,
                 n_steps, t_start, t_end, seed):
    """Simulated annealing with Gaussian position perturbation + slides.

    Two move types (50/50 mix):
      1. Gaussian displacement: (x + N(0, scale), y + N(0, scale))
         where scale = sqrt(T) * s_current. Allows free repositioning.
      2. Slide along rotational plane axis to random fraction of contact.
         Preserves contact structure.

    Moves that create overlaps are rejected. Energy = bounding box size.
    """
    np.random.seed(seed)
    init_xs = xs.copy()
    init_ys = ys.copy()
    best_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
    init_s = best_s
    best_xs = xs.copy()
    best_ys = ys.copy()
    current_s = best_s
    log_ratio = math.log(t_end / t_start)

    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        T = t_start * math.exp(log_ratio * frac)

        idx = np.random.randint(0, n)
        old_x, old_y = xs[idx], ys[idx]

        if np.random.random() < 0.5:
            # Move type 1: Gaussian position perturbation
            scale = math.sqrt(T) * max(current_s, 1.0) * 0.15
            move_x = old_x + np.random.standard_normal() * scale
            move_y = old_y + np.random.standard_normal() * scale

            # Reject if too far from any other square (prevents escape).
            # A square must remain within best_s of at least one neighbor.
            min_neighbor_dist = 1e30
            for j in range(n):
                if j == idx:
                    continue
                ddx = move_x - xs[j]
                ddy = move_y - ys[j]
                dist = math.sqrt(ddx * ddx + ddy * ddy)
                if dist < min_neighbor_dist:
                    min_neighbor_dist = dist
            if min_neighbor_dist > best_s:
                continue

            # Check overlaps with all other squares
            has_overlap = False
            for j in range(n):
                if j == idx:
                    continue
                if overlap_depth(move_x, move_y, cos_ts[idx], sin_ts[idx],
                                 xs[j], ys[j], cos_ts[j], sin_ts[j]) > 0.0:
                    has_overlap = True
                    break
            if has_overlap:
                continue
        else:
            # Move type 2: Slide along rotational plane axis
            k = np.random.randint(0, n_sweep)
            d = np.random.randint(0, 4)
            ck = sweep_cos[k]
            sk = sweep_sin[k]
            if d == 0: dx = ck; dy = sk
            elif d == 1: dx = -ck; dy = -sk
            elif d == 2: dx = -sk; dy = ck
            else: dx = sk; dy = -ck

            new_x, new_y = slide_diagonal(
                idx, dx, dy, 1e30, xs, ys, cos_ts, sin_ts, n)
            max_dist = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
            if max_dist < 1e-12:
                continue
            frac_slide = np.random.random()
            move_x = old_x + (new_x - old_x) * frac_slide
            move_y = old_y + (new_y - old_y) * frac_slide

        # Apply and measure
        xs[idx] = move_x
        ys[idx] = move_y
        new_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

        dE = new_s - current_s
        if dE <= 0 or (T > 1e-30 and np.random.random() < math.exp(-dE / T)):
            current_s = new_s
            if current_s < best_s:
                best_s = current_s
                for i in range(n):
                    best_xs[i] = xs[i]
                    best_ys[i] = ys[i]
        else:
            xs[idx] = old_x
            ys[idx] = old_y

    # Restore best and validate no overlaps
    for i in range(n):
        xs[i] = best_xs[i]
        ys[i] = best_ys[i]

    # Final overlap check — if overlaps crept in, restore initial state
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_depth(xs[i], ys[i], cos_ts[i], sin_ts[i],
                             xs[j], ys[j], cos_ts[j], sin_ts[j]) > 1e-8:
                for k in range(n):
                    xs[k] = init_xs[k]
                    ys[k] = init_ys[k]
                return init_s

    return best_s


@njit(cache=True)
def _shrink_and_resolve(xs, ys, cos_ts, sin_ts, n,
                        sweep_cos, sweep_sin, n_sweep,
                        shrink_factor):
    """Shrink all positions toward center, then resolve via sweeps.

    This can break through local minima by globally compressing the
    packing and then letting the exhaustive sweep find a new equilibrium.
    """
    # Compute center
    cx = 0.0
    cy = 0.0
    for i in range(n):
        cx += xs[i]
        cy += ys[i]
    cx /= n
    cy /= n

    # Shrink toward center
    for i in range(n):
        xs[i] = cx + (xs[i] - cx) * shrink_factor
        ys[i] = cy + (ys[i] - cy) * shrink_factor

    # Check for overlaps introduced by shrinking
    max_overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = overlap_depth(xs[i], ys[i], cos_ts[i], sin_ts[i],
                              xs[j], ys[j], cos_ts[j], sin_ts[j])
            if d > max_overlap:
                max_overlap = d

    return max_overlap


# ===================================================================
# Feasibility-style compression (random plane slides + exhaustive rigidity sweep)
# ===================================================================

@njit(cache=True)
def _feasibility_compress(xs, ys, cos_ts, sin_ts, n,
                          sweep_cos, sweep_sin, n_sweep,
                          n_steps, seed):
    """Feasibility-style compression for refinement.

    Random slides within a bounding-box wall constraint, then an
    exhaustive bbox-improving sweep. The whole random sequence is
    evaluated as a unit: only kept if the net result improves the bbox.
    """
    np.random.seed(seed)
    init_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

    # Save initial state
    save_xs = xs.copy()
    save_ys = ys.copy()

    # Use current bbox as wall constraint (with small margin to allow
    # rearrangement without explosion)
    x_min, x_max, y_min, y_max = xs[0], xs[0], ys[0], ys[0]
    for i in range(n):
        h = 0.5 * (abs(cos_ts[i]) + abs(sin_ts[i]))
        if xs[i] - h < x_min: x_min = xs[i] - h
        if xs[i] + h > x_max: x_max = xs[i] + h
        if ys[i] - h < y_min: y_min = ys[i] - h
        if ys[i] + h > y_max: y_max = ys[i] + h
    margin = (x_max - x_min) * 0.05
    wall_lo = x_min - margin
    wall_hi_x = x_max + margin
    wall_hi_y = y_max + margin
    wall_lo_y = y_min - margin

    # Random exploration: slides within walls
    for step in range(n_steps):
        idx = np.random.randint(0, n)
        k = np.random.randint(0, n_sweep)
        ck = sweep_cos[k]
        sk = sweep_sin[k]
        d = np.random.randint(0, 4)
        if d == 0: dx = ck;  dy = sk
        elif d == 1: dx = -ck; dy = -sk
        elif d == 2: dx = -sk; dy = ck
        else: dx = sk;  dy = -ck

        h_i = 0.5 * (abs(cos_ts[idx]) + abs(sin_ts[idx]))
        t_max = 1e30
        if dx > 1e-15:
            t_max = min(t_max, (wall_hi_x - h_i - xs[idx]) / dx)
        elif dx < -1e-15:
            t_max = min(t_max, (wall_lo + h_i - xs[idx]) / dx)
        if dy > 1e-15:
            t_max = min(t_max, (wall_hi_y - h_i - ys[idx]) / dy)
        elif dy < -1e-15:
            t_max = min(t_max, (wall_lo_y + h_i - ys[idx]) / dy)

        if t_max <= 1e-12:
            continue

        new_x, new_y = slide_diagonal(
            idx, dx, dy, t_max, xs, ys, cos_ts, sin_ts, n)
        if abs(new_x - xs[idx]) > 1e-12 or abs(new_y - ys[idx]) > 1e-12:
            xs[idx] = new_x
            ys[idx] = new_y

    # Exhaustive sweep: accept moves that don't worsen bbox.
    # Neutral moves (same bbox) are accepted because they may enable
    # dependent improvements later in the same pass. Once a square
    # accepts a neutral or improving move, skip its remaining directions
    # to avoid immediately undoing it.
    for sweep in range(50):
        any_improved = False
        for idx in range(n):
            moved_this_square = False
            for k in range(n_sweep):
                if moved_this_square:
                    break
                ck = sweep_cos[k]
                sk = sweep_sin[k]
                for d in range(4):
                    if d == 0: dx = ck;  dy = sk
                    elif d == 1: dx = -ck; dy = -sk
                    elif d == 2: dx = -sk; dy = ck
                    else: dx = sk;  dy = -ck

                    new_x, new_y = slide_diagonal(
                        idx, dx, dy, 1e30, xs, ys, cos_ts, sin_ts, n)
                    if (abs(new_x - xs[idx]) > 1e-12 or
                            abs(new_y - ys[idx]) > 1e-12):
                        old_x, old_y = xs[idx], ys[idx]
                        old_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
                        xs[idx] = new_x
                        ys[idx] = new_y
                        new_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
                        if new_s < old_s - 1e-14:
                            any_improved = True
                            moved_this_square = True
                            break
                        elif new_s > old_s + 1e-14:
                            xs[idx] = old_x
                            ys[idx] = old_y
                        else:
                            # Neutral: keep it, move on to next square
                            moved_this_square = True
                            break
        if not any_improved:
            break

    final_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

    # Only keep if net improvement
    if final_s >= init_s - 1e-14:
        for i in range(n):
            xs[i] = save_xs[i]
            ys[i] = save_ys[i]
        return init_s

    return final_s


# ===================================================================
# Public refinement API (called from sqpack.solver)
# ===================================================================

def refine_with_slides(n, xs, ys, thetas, rounds=100, sa_steps=500000,
                       sa_t_start=0.5, sa_t_end=1e-7, verbose=False):
    """Refine a packing using multi-scale SA with random partial slides.

    Two phases per round:
      Coarse: high temperature (sa_t_start -> sa_t_start/10), 30% of steps
              Explores large rearrangements.
      Fine:   low temperature (sa_t_start/10 -> sa_t_end), 70% of steps
              Fine-tunes positions.

    Does not change rotation angles.
    Returns (s, xs_list, ys_list, thetas_list).
    """
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    thetas = np.array(thetas, dtype=np.float64)
    cos_ts = np.cos(thetas)
    sin_ts = np.sin(thetas)

    sweep_cos, sweep_sin = get_sweep_planes(cos_ts, sin_ts, n)
    n_sweep = len(sweep_cos)

    best_s = float(bounding_box_size(xs, ys, cos_ts, sin_ts, n))
    best_xs = xs.copy()
    best_ys = ys.copy()

    coarse_steps = max(1, int(sa_steps * 0.3))
    fine_steps = sa_steps - coarse_steps
    t_mid = sa_t_start / 10.0

    for round_i in range(rounds):
        xs[:] = best_xs
        ys[:] = best_ys

        seed = int(time.time() * 1000 + round_i) % (2**31)

        # Coarse phase: high temperature, large rearrangements
        _sa_optimize(xs, ys, cos_ts, sin_ts, n,
                     sweep_cos, sweep_sin, n_sweep,
                     coarse_steps, sa_t_start, t_mid, seed)

        # Fine phase: low temperature, settle positions
        seed2 = (seed + 1) % (2**31)
        s_sa = float(_sa_optimize(xs, ys, cos_ts, sin_ts, n,
                                  sweep_cos, sweep_sin, n_sweep,
                                  fine_steps, t_mid, sa_t_end, seed2))

        if s_sa < best_s - 1e-14:
            best_s = s_sa
            best_xs[:] = xs
            best_ys[:] = ys
            if verbose:
                print(f"    slide refine round {round_i+1}: "
                      f"s={best_s:.10f}")

    return best_s, best_xs.tolist(), best_ys.tolist(), thetas.tolist()


def _try_angle_combination(n, xs_orig, ys_orig, thetas_orig, angle_groups,
                           candidate_angles, explode_factor, cx, cy,
                           n_steps, slide_rounds):
    """Try one combination of angles: explode, set angles, compress.

    Returns (s, xs_list, ys_list, thetas_list) or None if overlaps after explode.
    """
    xs_try = xs_orig.copy()
    ys_try = ys_orig.copy()
    thetas_try = thetas_orig.copy()

    # Explode
    for i in range(n):
        xs_try[i] = cx + (xs_try[i] - cx) * explode_factor
        ys_try[i] = cy + (ys_try[i] - cy) * explode_factor

    # Update rotations per group
    for g, (mask, angle) in enumerate(zip(angle_groups, candidate_angles)):
        for i in range(n):
            if mask[i]:
                thetas_try[i] = angle

    cos_try = np.cos(thetas_try)
    sin_try = np.sin(thetas_try)

    # Check no overlaps after explode + angle change
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_depth(xs_try[i], ys_try[i], cos_try[i], sin_try[i],
                             xs_try[j], ys_try[j], cos_try[j], sin_try[j]) > 1e-8:
                return None

    # Compress
    sweep_cos, sweep_sin = get_sweep_planes(cos_try, sin_try, n)
    n_sweep_try = len(sweep_cos)

    best_try_s = 1e30
    best_try_xs = xs_try.copy()
    best_try_ys = ys_try.copy()
    for r in range(slide_rounds):
        xs_round = best_try_xs.copy()
        ys_round = best_try_ys.copy()
        seed = int(time.time() * 1000 + r) % (2**31)
        s_round = float(_feasibility_compress(
            xs_round, ys_round, cos_try, sin_try, n,
            sweep_cos, sweep_sin, n_sweep_try, n_steps, seed))
        if s_round < best_try_s:
            best_try_s = s_round
            best_try_xs[:] = xs_round
            best_try_ys[:] = ys_round

    # Validate no overlaps in the result
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_depth(float(best_try_xs[i]), float(best_try_ys[i]),
                             float(cos_try[i]), float(sin_try[i]),
                             float(best_try_xs[j]), float(best_try_ys[j]),
                             float(cos_try[j]), float(sin_try[j])) > 1e-8:
                return None

    return (best_try_s, best_try_xs.tolist(), best_try_ys.tolist(),
            thetas_try.tolist())


def refine_with_angle_bisection(n, xs, ys, thetas, explode_factor=1.05,
                                bisection_steps=8, n_steps=2000,
                                slide_rounds=30, verbose=False):
    """Refine by perturbing all non-zero rotation angles.

    Each non-zero angle group gets its own perturbation range.
    Searches over the joint angle space, narrowing each dimension
    independently per bisection step.

    Returns (s, xs_list, ys_list, thetas_list).
    """
    xs_orig = np.array(xs, dtype=np.float64)
    ys_orig = np.array(ys, dtype=np.float64)
    thetas_orig = np.array(thetas, dtype=np.float64)

    # Identify all distinct non-zero rotation angles and their masks
    unique_angles = sorted(set(round(float(t), 10) for t in thetas_orig))
    nonzero_angles = [a for a in unique_angles if a > 1e-6]
    if not nonzero_angles:
        s = float(bounding_box_size(xs_orig, ys_orig,
                                    np.cos(thetas_orig), np.sin(thetas_orig), n))
        return s, xs, ys, thetas

    # Build per-group masks
    angle_groups = []  # list of boolean arrays
    for base_angle in nonzero_angles:
        mask = np.array([abs(float(t) - base_angle) < 1e-6 for t in thetas_orig])
        angle_groups.append(mask)
    n_groups = len(nonzero_angles)

    cx = float(np.mean(xs_orig))
    cy = float(np.mean(ys_orig))

    best_s = float(bounding_box_size(xs_orig, ys_orig,
                                     np.cos(thetas_orig), np.sin(thetas_orig), n))
    best_xs = list(xs)
    best_ys = list(ys)
    best_thetas = list(thetas)

    # Per-group search ranges: angle ± delta
    delta = 0.1
    ranges_lo = [max(1e-6, a - delta) for a in nonzero_angles]
    ranges_hi = [min(math.pi / 4 - 1e-6, a + delta) for a in nonzero_angles]
    best_angles = list(nonzero_angles)

    if verbose:
        for g in range(n_groups):
            print(f"    angle group {g+1}: base={nonzero_angles[g]:.6f}, "
                  f"range=[{ranges_lo[g]:.6f}, {ranges_hi[g]:.6f}]")

    # Number of samples per dimension per bisection step
    n_samples = 5

    for bis_step in range(bisection_steps):
        # Build candidate grids per group
        grids = [np.linspace(ranges_lo[g], ranges_hi[g], n_samples)
                 for g in range(n_groups)]

        step_best_s = 1e30
        step_best_angles = list(best_angles)
        step_best_data = None

        if n_groups == 1:
            # 1D search: 5 candidates
            for a0 in grids[0]:
                result = _try_angle_combination(
                    n, xs_orig, ys_orig, thetas_orig, angle_groups,
                    [a0], explode_factor, cx, cy, n_steps, slide_rounds)
                if result is not None and result[0] < step_best_s:
                    step_best_s = result[0]
                    step_best_angles = [float(a0)]
                    step_best_data = result
        elif n_groups == 2:
            # 2D search: 5x5 = 25 candidates
            for a0 in grids[0]:
                for a1 in grids[1]:
                    result = _try_angle_combination(
                        n, xs_orig, ys_orig, thetas_orig, angle_groups,
                        [a0, a1], explode_factor, cx, cy,
                        n_steps, slide_rounds)
                    if result is not None and result[0] < step_best_s:
                        step_best_s = result[0]
                        step_best_angles = [float(a0), float(a1)]
                        step_best_data = result
        else:
            # 3+ groups: search each dimension independently to avoid
            # combinatorial explosion. Fix other angles at current best
            # and sweep one at a time.
            for g in range(n_groups):
                for a_g in grids[g]:
                    cand = list(best_angles)
                    cand[g] = float(a_g)
                    result = _try_angle_combination(
                        n, xs_orig, ys_orig, thetas_orig, angle_groups,
                        cand, explode_factor, cx, cy, n_steps, slide_rounds)
                    if result is not None and result[0] < step_best_s:
                        step_best_s = result[0]
                        step_best_angles = list(cand)
                        step_best_data = result

        if step_best_data is not None and step_best_s < best_s - 1e-14:
            best_s = step_best_s
            best_xs = step_best_data[1]
            best_ys = step_best_data[2]
            best_thetas = step_best_data[3]
            best_angles = list(step_best_angles)
            xs_orig = np.array(best_xs, dtype=np.float64)
            ys_orig = np.array(best_ys, dtype=np.float64)
            thetas_orig = np.array(best_thetas, dtype=np.float64)
            if verbose:
                angles_str = ", ".join(f"{a:.8f}" for a in step_best_angles)
                print(f"    bisection step {bis_step+1}: "
                      f"angles=[{angles_str}] -> s={best_s:.10f}")

        # Narrow each range around its best angle
        for g in range(n_groups):
            spread = (ranges_hi[g] - ranges_lo[g]) / 4
            ranges_lo[g] = max(1e-6, step_best_angles[g] - spread)
            ranges_hi[g] = min(math.pi / 4 - 1e-6, step_best_angles[g] + spread)

    return best_s, best_xs, best_ys, best_thetas


# ===================================================================
# Polish: iterative theta scan with convergence
# ===================================================================

def _scan_theta(n, xs_base, ys_base, thetas_base, rotated, cx, cy,
                lo_deg, hi_deg, theta_points, explode_factor,
                sa_rounds, sa_steps, sa_t_start, sa_t_end, verbose):
    """Scan theta over [lo_deg, hi_deg]. Returns (best_s, xs, ys, thetas, best_deg)."""
    candidates = np.linspace(lo_deg, hi_deg, theta_points)
    best_s = float('inf')
    best_xs = None
    best_ys = None
    best_thetas = None
    best_deg = (lo_deg + hi_deg) / 2

    for deg in candidates:
        theta_new = math.radians(deg)
        xs_try = xs_base.copy()
        ys_try = ys_base.copy()
        thetas_try = thetas_base.copy()

        for i in range(n):
            xs_try[i] = cx + (xs_try[i] - cx) * explode_factor
            ys_try[i] = cy + (ys_try[i] - cy) * explode_factor
        for i in range(n):
            if rotated[i]:
                thetas_try[i] = theta_new
        cos_try = np.cos(thetas_try)
        sin_try = np.sin(thetas_try)

        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                if overlap_depth(float(xs_try[i]), float(ys_try[i]),
                                 float(cos_try[i]), float(sin_try[i]),
                                 float(xs_try[j]), float(ys_try[j]),
                                 float(cos_try[j]), float(sin_try[j])) > 0:
                    has_overlap = True
                    break
            if has_overlap:
                break
        if has_overlap:
            continue

        s_ref, xs_ref, ys_ref, thetas_ref = refine_with_slides(
            n, xs_try.tolist(), ys_try.tolist(), thetas_try.tolist(),
            rounds=sa_rounds, sa_steps=sa_steps,
            sa_t_start=sa_t_start, sa_t_end=sa_t_end)

        if verbose:
            _KB = KNOWN_BEST
            gap_str = ""
            if n in _KB:
                gap = 100 * (s_ref - _KB[n]) / _KB[n]
                gap_str = f"  gap={gap:.6f}%"
            marker = " ***" if s_ref < best_s else ""
            print(f"      theta={deg:8.4f} deg  s={s_ref:.10f}{gap_str}{marker}")

        if s_ref < best_s - 1e-14:
            best_s = s_ref
            best_xs = xs_ref
            best_ys = ys_ref
            best_thetas = thetas_ref
            best_deg = deg

    return best_s, best_xs, best_ys, best_thetas, best_deg


def polish_theta(n, xs, ys, thetas, theta_range=1.0, theta_points=11,
                 explode_factor=1.05, sa_rounds=20, sa_steps=200000,
                 sa_t_start=0.5, sa_t_end=1e-7, convergence_tol=0.01,
                 verbose=False):
    """Polish a packing by iteratively narrowing the theta scan.

    Each iteration scans theta_points values, finds the best, then
    narrows the range by half and repeats until the range is below
    convergence_tol degrees.

    Returns (best_s, best_xs, best_ys, best_thetas).
    """
    xs_base = np.array(xs, dtype=np.float64)
    ys_base = np.array(ys, dtype=np.float64)
    thetas_base = np.array(thetas, dtype=np.float64)
    rotated = np.array([t > 0.01 for t in thetas_base])

    if not rotated.any():
        s = float(bounding_box_size(xs_base, ys_base,
                                    np.cos(thetas_base), np.sin(thetas_base), n))
        return s, xs, ys, thetas

    base_theta = float(thetas_base[rotated][0])
    base_deg = math.degrees(base_theta)
    cx = float(xs_base.mean())
    cy = float(ys_base.mean())

    s_orig = float(bounding_box_size(xs_base, ys_base,
                                     np.cos(thetas_base), np.sin(thetas_base), n))
    best_s = s_orig
    best_xs = list(xs)
    best_ys = list(ys)
    best_thetas = list(thetas)
    best_deg = base_deg

    lo_deg = max(0.1, base_deg - theta_range)
    hi_deg = min(44.9, base_deg + theta_range)

    while hi_deg - lo_deg > convergence_tol:
        s_iter, xs_iter, ys_iter, thetas_iter, deg_iter = _scan_theta(
            n, xs_base, ys_base, thetas_base, rotated, cx, cy,
            lo_deg, hi_deg, theta_points, explode_factor,
            sa_rounds, sa_steps, sa_t_start, sa_t_end, verbose)

        if xs_iter is not None and s_iter < best_s - 1e-14:
            best_s = s_iter
            best_xs = xs_iter
            best_ys = ys_iter
            best_thetas = thetas_iter
            best_deg = deg_iter
            xs_base = np.array(best_xs, dtype=np.float64)
            ys_base = np.array(best_ys, dtype=np.float64)
            thetas_base = np.array(best_thetas, dtype=np.float64)
            cx = float(xs_base.mean())
            cy = float(ys_base.mean())

        if verbose:
            _KB = KNOWN_BEST
            gap_str = ""
            if n in _KB:
                gap = 100 * (best_s - _KB[n]) / _KB[n]
                gap_str = f" (gap: {gap:.6f}%)"
            print(f"    >> Polish: theta={best_deg:.6f} deg, "
                  f"s={best_s:.10f}{gap_str}")

        half_range = (hi_deg - lo_deg) / 4
        lo_deg = max(0.1, best_deg - half_range)
        hi_deg = min(44.9, best_deg + half_range)

    return best_s, best_xs, best_ys, best_thetas


# ===================================================================
# Python orchestration
# ===================================================================

def get_sweep_planes(cos_ts, sin_ts, n):
    """Extract unique rotation planes from placed squares."""
    sweep_cos = []
    sweep_sin = []
    for i in range(n):
        ci, si = float(cos_ts[i]), float(sin_ts[i])
        is_dup = False
        for j in range(len(sweep_cos)):
            if abs(ci - sweep_cos[j]) < 1e-9 and abs(si - sweep_sin[j]) < 1e-9:
                is_dup = True
                break
        if not is_dup:
            sweep_cos.append(ci)
            sweep_sin.append(si)
    return np.array(sweep_cos), np.array(sweep_sin)


def optimize(n, s, xs, ys, thetas, rounds=100, sa_steps=100000,
             verbose=False):
    """Run multi-strategy optimization on an existing packing."""
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    thetas = np.array(thetas, dtype=np.float64)
    cos_ts = np.cos(thetas)
    sin_ts = np.sin(thetas)

    sweep_cos, sweep_sin = get_sweep_planes(cos_ts, sin_ts, n)
    n_sweep = len(sweep_cos)

    if verbose:
        print(f"Optimizing: n={n}, starting s={s:.10f}")
        print(f"  Sweep planes: {n_sweep}")
        if n in KNOWN_BEST:
            gap = 100 * (s - KNOWN_BEST[n]) / KNOWN_BEST[n]
            print(f"  Known optimal: {KNOWN_BEST[n]:.10f} (gap: {gap:.4f}%)")
        print()

    best_s = bounding_box_size(xs, ys, cos_ts, sin_ts, n)
    best_xs = xs.copy()
    best_ys = ys.copy()

    t0 = time.time()

    for round_i in range(rounds):
        # Strategy 1: Bbox-constrained sweep (only accept moves that shrink bbox)
        moves = _bbox_constrained_sweep(xs, ys, cos_ts, sin_ts, n,
                                        sweep_cos, sweep_sin, n_sweep, 100)
        s_after_sweep = bounding_box_size(xs, ys, cos_ts, sin_ts, n)

        if s_after_sweep < best_s - 1e-14:
            best_s = s_after_sweep
            best_xs[:] = xs
            best_ys[:] = ys
            if verbose:
                gap_str = ""
                if n in KNOWN_BEST:
                    gap = 100 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
                    gap_str = f" (gap: {gap:.4f}%)"
                print(f"  Round {round_i+1}: sweep {moves} moves -> "
                      f"s={best_s:.10f}{gap_str}")

        # Strategy 2: Coordinate descent (fine moves, accept only improvements)
        step_size = max(0.001, 0.1 * (1.0 - round_i / max(1, rounds)))
        s_cd, n_improved = _coordinate_descent(
            xs, ys, cos_ts, sin_ts, n,
            sweep_cos, sweep_sin, n_sweep,
            n_steps=50, step_size=step_size)

        if s_cd < best_s - 1e-14:
            best_s = s_cd
            best_xs[:] = xs
            best_ys[:] = ys
            if verbose:
                gap_str = ""
                if n in KNOWN_BEST:
                    gap = 100 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
                    gap_str = f" (gap: {gap:.4f}%)"
                print(f"  Round {round_i+1}: coord descent {n_improved} "
                      f"improvements -> s={best_s:.10f}{gap_str}")

        # Strategy 3: SA with random partial slides
        sa_t_start = max(0.01, 0.1 * (1.0 - round_i / max(1, rounds)))
        sa_t_end = 1e-6
        seed = int(time.time() * 1000 + round_i) % (2**31)
        s_sa = _sa_optimize(xs, ys, cos_ts, sin_ts, n,
                            sweep_cos, sweep_sin, n_sweep,
                            sa_steps, sa_t_start, sa_t_end, seed)

        if s_sa < best_s - 1e-14:
            best_s = s_sa
            best_xs[:] = xs
            best_ys[:] = ys
            if verbose:
                gap_str = ""
                if n in KNOWN_BEST:
                    gap = 100 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
                    gap_str = f" (gap: {gap:.4f}%)"
                print(f"  Round {round_i+1}: SA -> "
                      f"s={best_s:.10f}{gap_str}")
        else:
            # SA didn't improve — restore best
            xs[:] = best_xs
            ys[:] = best_ys

        # Strategy 4: Shrink and resolve (every 10 rounds)
        if (round_i + 1) % 10 == 0:
            xs_try = best_xs.copy()
            ys_try = best_ys.copy()
            # Try shrinking by progressively smaller factors
            for shrink in [0.999, 0.998, 0.995, 0.99]:
                xs_try[:] = best_xs
                ys_try[:] = best_ys
                max_ol = _shrink_and_resolve(
                    xs_try, ys_try, cos_ts, sin_ts, n,
                    sweep_cos, sweep_sin, n_sweep, shrink)
                if max_ol < 1e-10:
                    # No overlaps from shrink — sweep to settle
                    _bbox_constrained_sweep(xs_try, ys_try, cos_ts, sin_ts, n,
                                            sweep_cos, sweep_sin, n_sweep, 100)
                    s_shrink = bounding_box_size(
                        xs_try, ys_try, cos_ts, sin_ts, n)
                    if s_shrink < best_s - 1e-14:
                        best_s = s_shrink
                        best_xs[:] = xs_try
                        best_ys[:] = ys_try
                        xs[:] = xs_try
                        ys[:] = ys_try
                        if verbose:
                            gap_str = ""
                            if n in KNOWN_BEST:
                                gap = 100 * (best_s - KNOWN_BEST[n]) / KNOWN_BEST[n]
                                gap_str = f" (gap: {gap:.4f}%)"
                            print(f"  Round {round_i+1}: shrink({shrink}) -> "
                                  f"s={best_s:.10f}{gap_str}")
                        break

        # Periodic status
        if verbose and (round_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... round {round_i+1}/{rounds}, "
                  f"best s={best_s:.10f}, {elapsed:.1f}s")

    elapsed = time.time() - t0

    # Validate
    max_d = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = overlap_depth(best_xs[i], best_ys[i], cos_ts[i], sin_ts[i],
                              best_xs[j], best_ys[j], cos_ts[j], sin_ts[j])
            if d > max_d:
                max_d = d
    if max_d > 1e-8:
        print(f"WARNING: max overlap {max_d:.6e}")

    return best_s, best_xs.tolist(), best_ys.tolist(), thetas.tolist(), elapsed


