"""sqpack: pack n unit squares into the smallest enclosing square."""

from .schema import Result, SchemaError, SCHEMA_VERSION
from .io import load_result, save_result
from .known_best import KNOWN_BEST
from .solver import solve
from .refinement import (
    refine_with_slides,
    refine_with_angle_bisection,
    polish_theta,
    optimize as refine,
    polish_theta as polish,
)
from .visualization import plot_packing

__version__ = "0.1.0"

__all__ = [
    "Result",
    "SchemaError",
    "SCHEMA_VERSION",
    "load_result",
    "save_result",
    "KNOWN_BEST",
    "solve",
    "refine",
    "polish",
    "refine_with_slides",
    "refine_with_angle_bisection",
    "polish_theta",
    "plot_packing",
    "__version__",
]
