# sqpack Python API

Everything below imports from the top-level `sqpack` package.

## `solve(n, **kwargs) -> Result`

Run the full feasibility-search + refine + polish pipeline. Returns a
`Result` dataclass.

```python
from sqpack import solve, save_result

result = solve(11, time_limit=300, verbose=True)
save_result(result, "n11.json")
```

Notable keyword arguments (all optional):

- `n_compress_steps=1000`, `compress_restarts=1`
- `n_rotations=2`, `numrotate=None`
- `s_start=None` (defaults to `ceil(sqrt(n))`)
- `cohort_size=10`, `angle_power=8`
- `seed=None`, `verbose=False`
- `time_limit=None` (seconds), `workers=None`
- `on_new_best=callable(s, xs, ys, thetas, n_total)` for streaming improvements
- `enable_polish=True`
- A handful of `refine_*` and `polish_*` knobs that mirror the CLI flags.

## `refine(n, s, xs, ys, thetas, ...) -> (best_s, xs, ys, thetas, elapsed)`

Aliased to `optimize`. Applies multi-axis sweep + coordinate descent +
SA + global shrink to push an existing packing tighter.

```python
from sqpack import refine, load_result, save_result

r = load_result("n11.json")
best_s, xs, ys, thetas, elapsed = refine(
    r.n, r.s, r.xs, r.ys, r.thetas, rounds=200, sa_steps=200000, verbose=True,
)
```

## `polish(n, xs, ys, thetas, ...)`

Aliased to `polish_theta`. Per-square theta scan with SA at each angle.
Used as the final stage of `solve()`.

## `load_result(path) -> Result`

Reads v1 JSON or migrates a legacy flat file in memory. Does not write.

## `save_result(result, path, **kwargs) -> path`

Writes a v1 JSON file. Accepts a `Result` or a pre-validated v1 dict.
Optional keyword args: `solver`, `history`, `top_10`, `extra_metadata`.

## `plot_packing(n, s, xs, ys, thetas, ...)`

Render a packing with matplotlib. Optional `filename=` saves a PNG;
`show=True` opens an interactive window.

## `KNOWN_BEST`

`dict[int, float]` of the best-known `s(n)` values from the David
Ellsworth catalog.

```python
from sqpack import KNOWN_BEST
print(KNOWN_BEST[11])  # 3.87708359002281
```

## `Result`

Dataclass returned by `solve` and `load_result`.

```python
@dataclass
class Result:
    n: int
    s: float
    xs: list      # length n, floats
    ys: list      # length n, floats
    thetas: list  # length n, radians
    s_start: float = 0.0
    n_total: int = 0
    n_feasible: int = 0
    n_nontrivial: int = 0
    elapsed: float = 0.0
    metadata: dict = {}
    history: list = []
    top_10: list = []
```

## `SchemaError`

Raised by `load_result` / `validate_v1` when a payload cannot be parsed.
