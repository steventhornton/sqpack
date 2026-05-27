# AGENTS.md

LLM-facing guide for working inside the `sqpack` repository. Read this
before doing non-trivial work.

## Problem

Pack `n` unit squares (each freely rotatable) into the smallest possible
enclosing axis-aligned square of side `s`. The goal is to find
configurations that match or beat the best-known `s(n)` values
catalogued by David Ellsworth (mirrored in `sqpack/data/known_best.json`,
which itself mirrors Erich Friedman's table).

A run is "successful" when its reported `s` matches `KNOWN_BEST[n]` to
within polish noise (~1e-6). Beating a known best is the research goal.

## Installation

Requires Python 3.10+ in a venv (Homebrew Python blocks system-wide pip
installs per PEP 668).

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                              # local development
pip install git+https://github.com/.../sqpack.git   # from GitHub
```

`uv venv && uv pip install -e .` works too and is faster.

## Running the CLI

```
sqpack --help                 # top-level help, lists subcommands
sqpack solve --help           # every flag for a subcommand
sqpack refine --help
sqpack render --help
sqpack info --help
```

Typical n=11 run:

```
sqpack solve 11 --rotations 2 --numrotate 5 --time 300 -v
sqpack info   output/sqpack_n_11_*.json
sqpack render output/sqpack_n_11_*.json -o n11.png
sqpack refine output/sqpack_n_11_*.json --rounds 200 -v
```

`docs/cli.md` mirrors the `--help` output with worked examples.

## Pipeline at a glance

1. **Cohort loop** at fixed `s_start`. Repeats until time budget runs out.
2. **Drop**: place each square via rest-on-contact from a random
   direction.
3. **Compress**: random slides along the trial's rotation planes,
   followed by an exhaustive rigidity sweep.
4. **Refine**: SA slides + angle bisection on the cohort's smallest
   realised bbox.
5. **Polish**: iterative theta scan around the current best angle.
6. Result is serialized to JSON v1 via `sqpack.io.save_result`.

Full walk-through in `docs/internals.md`. JSON layout in `docs/output-schema.md`.

## Public Python API

```python
from sqpack import (
    solve, refine, polish,
    refine_with_slides, refine_with_angle_bisection, polish_theta,
    load_result, save_result, plot_packing,
    KNOWN_BEST, Result, SchemaError, SCHEMA_VERSION,
)
```

`refine` is an alias for `optimize` (the multi-strategy refinement in
`sqpack/refinement.py`). `polish` is an alias for `polish_theta`. Full
reference in `docs/python-api.md`.

## Available skills

- `.claude/skills/run-experiment/` — kick off a `sqpack solve` for a
  given `n` with sensible defaults.
- `.claude/skills/analyze-results/` — inspect, summarize, render, and
  compare a result JSON against `KNOWN_BEST`.

## Development guidelines

### Repo layout

- Active source: `sqpack/`.
- Documentation: `docs/`.
- Research history (preserved but not in this repo): handled separately.

### Don't touch without measuring

- Anything Numba-decorated (`@njit`) in `sqpack/geometry.py`,
  `sqpack/solver.py`, and `sqpack/refinement.py`. These are on the
  trial inner loop. Run a microbenchmark before changing.
- `sqpack/data/known_best.json` — the table of literature values. Edit
  with care; cite the source if you add entries.
- The `numrotate` argument structure (`--rotations R` requires
  `--numrotate` with `R-1` comma-separated values). External sweep
  invocations depend on this exact shape.

### House style

- Public functions go through `sqpack.io.save_result` / `load_result`
  for JSON I/O — never write JSON directly. The v1 schema is the only
  on-disk format.
- New CLI flags belong in `sqpack/cli/<subcommand>.py` and need a row
  in `docs/cli.md`.
- When `docs/internals.md` cites a file:line, the docs are the thing
  that goes stale, not the code. Update citations when you refactor.

### When uncertain

Cross-reference `docs/internals.md` against the source. The pipeline
documentation is the contract; if it's wrong, fix the doc rather than
contorting the code to match.
