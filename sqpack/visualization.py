"""
Visualization for square packing results.

Renders a packing as a matplotlib figure with rotated square patches
inside the enclosing container.
"""

import math

import numpy as np

from .known_best import KNOWN_BEST


def plot_packing(n, s, xs, ys, thetas, title_prefix="",
                 filename=None, show=True):
    """Plot a square packing.

    Args:
        n: Number of squares.
        s: Side length of the enclosing container.
        xs: List/array of x-coordinates (centers).
        ys: List/array of y-coordinates (centers).
        thetas: List/array of rotation angles.
        title_prefix: Optional prefix for the plot title.
        filename: If set, save PNG to this path.
        show: If True, display the plot interactively.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not xs:
        print("No valid packing to visualize.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    xs = np.array(xs)
    ys = np.array(ys)
    thetas = np.array(thetas)
    cos_ts = np.cos(thetas)
    sin_ts = np.sin(thetas)
    hs = 0.5 * (np.abs(cos_ts) + np.abs(sin_ts))

    # Shift so bounding box starts at origin
    x_shift = float(np.min(xs - hs))
    y_shift = float(np.min(ys - hs))
    xs = xs - x_shift
    ys = ys - y_shift

    # Container
    ax.add_patch(patches.Rectangle(
        (0, 0), s, s, linewidth=2,
        edgecolor='black', facecolor='lightyellow'))

    # Squares
    cmap = plt.cm.Set3(np.linspace(0, 1, max(n, 1)))
    for i in range(n):
        cx, cy = float(xs[i]), float(ys[i])
        ct, st = float(cos_ts[i]), float(sin_ts[i])
        blx = cx - 0.5 * ct + 0.5 * st
        bly = cy - 0.5 * st - 0.5 * ct
        rect = patches.Rectangle(
            (blx, bly), 1, 1, angle=math.degrees(thetas[i]),
            linewidth=1, edgecolor='black',
            facecolor=cmap[i % len(cmap)], alpha=0.7)
        ax.add_patch(rect)
        ax.text(cx, cy, str(i + 1), ha='center', va='center', fontsize=8)

    ax.set_xlim(-0.1, s + 0.1)
    ax.set_ylim(-0.1, s + 0.1)
    ax.set_aspect('equal')

    gap_str = ""
    if n in KNOWN_BEST:
        gap = 100 * (s - KNOWN_BEST[n]) / KNOWN_BEST[n]
        gap_str = f"  gap={gap:.2f}%"
    prefix = f"{title_prefix}: " if title_prefix else ""
    ax.set_title(f'{prefix}n={n}, s={s:.6f}{gap_str}')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    if show:
        plt.show()
    plt.close(fig)
