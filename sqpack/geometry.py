"""
Numba JIT-compiled geometry primitives for square packing.

Provides SAT-based overlap detection, analytical overlap intervals,
drop/slide operations, and bounding box computation for rotated unit squares.
"""

import math

from numba import njit

QUARTER_PI = math.pi / 4.0
SQRT2 = 1.4142135623730951


# ===================================================================
# Overlap detection (SAT)
# ===================================================================

@njit(cache=True)
def overlap_depth(xi, yi, ci, si, xj, yj, cj, sj):
    """SAT overlap depth between two unit squares.

    Takes precomputed cos/sin of angles. Returns penetration depth (>0 if
    overlapping, 0 otherwise).
    """
    dx = xj - xi
    dy = yj - yi
    cos_dt = cj * ci + sj * si
    sin_dt = sj * ci - cj * si
    H = 0.5 + 0.5 * (abs(cos_dt) + abs(sin_dt))
    p = max(abs(dx*ci + dy*si), abs(-dx*si + dy*ci),
            abs(dx*cj + dy*sj), abs(-dx*sj + dy*cj))
    d = H - p
    return d if d > 0.0 else 0.0


# ===================================================================
# Analytical overlap intervals along axes
#
# For two rotated unit squares, the SAT gives 4 separating axes.
# Each axis constrains the variable coordinate to an interval.
# The overlap region is the intersection of these 4 intervals.
# This gives O(1) exact contact-point computation per pair.
# ===================================================================

@njit(cache=True)
def overlap_y_interval(x_i, ci, si, xj, yj, cj, sj):
    """Compute the y_i interval where square i overlaps square j.

    Returns (lo, hi, valid). If valid=False, no overlap at any y_i.
    """
    dx = xj - x_i
    cos_dt = cj * ci + sj * si
    sin_dt = sj * ci - cj * si
    H = 0.5 + 0.5 * (abs(cos_dt) + abs(sin_dt))

    A0 = dx * ci + yj * si;   B0 = -si
    A1 = -dx * si + yj * ci;  B1 = -ci
    A2 = dx * cj + yj * sj;   B2 = -sj
    A3 = -dx * sj + yj * cj;  B3 = -cj

    lo = -1e30
    hi = 1e30
    for k in range(4):
        if k == 0: A = A0; B = B0
        elif k == 1: A = A1; B = B1
        elif k == 2: A = A2; B = B2
        else: A = A3; B = B3

        if abs(B) < 1e-15:
            if abs(A) >= H:
                return (0.0, 0.0, False)
        elif B > 0.0:
            v = (-H - A) / B
            if v > lo: lo = v
            v = (H - A) / B
            if v < hi: hi = v
        else:
            v = (H - A) / B
            if v > lo: lo = v
            v = (-H - A) / B
            if v < hi: hi = v

    if lo >= hi - 1e-12:
        return (0.0, 0.0, False)
    return (lo, hi, True)


@njit(cache=True)
def overlap_x_interval(y_i, ci, si, xj, yj, cj, sj):
    """Compute the x_i interval where square i overlaps square j.

    Returns (lo, hi, valid). If valid=False, no overlap at any x_i.
    """
    dy = yj - y_i
    cos_dt = cj * ci + sj * si
    sin_dt = sj * ci - cj * si
    H = 0.5 + 0.5 * (abs(cos_dt) + abs(sin_dt))

    A0 = xj * ci + dy * si;   B0 = -ci
    A1 = -xj * si + dy * ci;  B1 = si
    A2 = xj * cj + dy * sj;   B2 = -cj
    A3 = -xj * sj + dy * cj;  B3 = sj

    lo = -1e30
    hi = 1e30
    for k in range(4):
        if k == 0: A = A0; B = B0
        elif k == 1: A = A1; B = B1
        elif k == 2: A = A2; B = B2
        else: A = A3; B = B3

        if abs(B) < 1e-15:
            if abs(A) >= H:
                return (0.0, 0.0, False)
        elif B > 0.0:
            v = (-H - A) / B
            if v > lo: lo = v
            v = (H - A) / B
            if v < hi: hi = v
        else:
            v = (H - A) / B
            if v > lo: lo = v
            v = (-H - A) / B
            if v < hi: hi = v

    if lo >= hi - 1e-12:
        return (0.0, 0.0, False)
    return (lo, hi, True)


@njit(cache=True)
def overlap_t_interval(xi, yi, dir_x, dir_y, ci, si, xj, yj, cj, sj):
    """Overlap t-interval when square i moves along direction (dir_x, dir_y).

    Square i at (xi + t*dir_x, yi + t*dir_y). Returns (lo, hi, valid).
    """
    ddx = xj - xi
    ddy = yj - yi
    cos_dt = cj * ci + sj * si
    sin_dt = sj * ci - cj * si
    H = 0.5 + 0.5 * (abs(cos_dt) + abs(sin_dt))

    A0 = ddx * ci + ddy * si;      B0 = -(dir_x * ci + dir_y * si)
    A1 = -ddx * si + ddy * ci;     B1 = dir_x * si - dir_y * ci
    A2 = ddx * cj + ddy * sj;      B2 = -(dir_x * cj + dir_y * sj)
    A3 = -ddx * sj + ddy * cj;     B3 = dir_x * sj - dir_y * cj

    lo = -1e30
    hi = 1e30
    for k in range(4):
        if k == 0: A = A0; B = B0
        elif k == 1: A = A1; B = B1
        elif k == 2: A = A2; B = B2
        else: A = A3; B = B3

        if abs(B) < 1e-15:
            if abs(A) >= H:
                return (0.0, 0.0, False)
        elif B > 0.0:
            v = (-H - A) / B
            if v > lo: lo = v
            v = (H - A) / B
            if v < hi: hi = v
        else:
            v = (H - A) / B
            if v > lo: lo = v
            v = (-H - A) / B
            if v < hi: hi = v

    if lo >= hi - 1e-12:
        return (0.0, 0.0, False)
    return (lo, hi, True)


# ===================================================================
# Drop and slide operations
# ===================================================================

@njit(cache=True)
def find_drop_y(x, ci, si, xs, ys, cos_ts, sin_ts, count, exclude):
    """Find resting y when dropping a square from above.

    Checks against the first `count` squares, excluding index `exclude`.
    Returns the y coordinate where the square rests (contact point).
    """
    h = 0.5 * (abs(ci) + abs(si))
    rest_y = h
    for j in range(count):
        if j == exclude:
            continue
        lo, y_hi, valid = overlap_y_interval(
            x, ci, si, xs[j], ys[j], cos_ts[j], sin_ts[j])
        if valid and y_hi > rest_y:
            rest_y = y_hi
    return rest_y


@njit(cache=True)
def slide_diagonal(idx, dir_x, dir_y, target_t, xs, ys, cos_ts, sin_ts, n):
    """Slide square idx along direction (dir_x, dir_y) by at most target_t.

    Returns (new_x, new_y). Stops at first collision.
    """
    ci = cos_ts[idx]
    si = sin_ts[idx]
    xi = xs[idx]
    yi = ys[idx]

    if abs(target_t) < 1e-12:
        return xi, yi

    moving_pos = target_t > 0
    limit_t = target_t

    for j in range(n):
        if j == idx:
            continue
        t_lo, t_hi, valid = overlap_t_interval(
            xi, yi, dir_x, dir_y, ci, si,
            xs[j], ys[j], cos_ts[j], sin_ts[j])
        if not valid:
            continue
        if moving_pos:
            if t_lo >= -1e-12 and t_lo < limit_t:
                limit_t = t_lo
        else:
            if t_hi <= 1e-12 and t_hi > limit_t:
                limit_t = t_hi

    return xi + limit_t * dir_x, yi + limit_t * dir_y


# ===================================================================
# Bounding box
# ===================================================================

@njit(cache=True)
def bounding_box(xs, ys, cos_ts, sin_ts, n):
    """Minimal axis-aligned bounding box. Returns (x_min, x_max, y_min, y_max)."""
    x_min = 1e30
    x_max = -1e30
    y_min = 1e30
    y_max = -1e30
    for i in range(n):
        h = 0.5 * (abs(cos_ts[i]) + abs(sin_ts[i]))
        v = xs[i] - h
        if v < x_min: x_min = v
        v = xs[i] + h
        if v > x_max: x_max = v
        v = ys[i] - h
        if v < y_min: y_min = v
        v = ys[i] + h
        if v > y_max: y_max = v
    return x_min, x_max, y_min, y_max


@njit(cache=True)
def bounding_box_size(xs, ys, cos_ts, sin_ts, n):
    """Side length of the minimal enclosing square."""
    x_min, x_max, y_min, y_max = bounding_box(xs, ys, cos_ts, sin_ts, n)
    dx = x_max - x_min
    dy = y_max - y_min
    return dx if dx > dy else dy
