# sqpack output JSON schema (v1.0)

`sqpack solve` and `sqpack refine` write a single JSON file. The shape below
is the canonical layout. `sqpack.io.load_result` accepts both this v1 layout
and the older flat layout (no `schema_version` key); legacy files are
migrated in memory on read, and re-emitted as v1 only when explicitly saved
via `sqpack.io.save_result`.

## Top-level keys

| Key              | Type      | Required | Description                                     |
| ---------------- | --------- | -------- | ----------------------------------------------- |
| `schema_version` | string    | yes      | Exactly `"1.0"`.                                |
| `n`              | int       | yes      | Number of unit squares.                         |
| `s`              | float     | yes      | Side length of the enclosing axis-aligned square. |
| `squares`        | array     | yes      | Length-`n`. Order is solver-internal.           |
| `metadata`       | object    | yes      | Run provenance, see below.                      |
| `history`        | array     | no       | Append-only stage records, see below.           |
| `top_10`         | array     | no       | Up to 10 best non-trivial packings ever seen.   |

### `squares[i]`

Each square is `{"x": float, "y": float, "theta": float}`. `theta` is in
radians. `(x, y)` is the center of the unit square. Bounds: the rotated
unit-square footprint lies inside `[0, s] x [0, s]`.

### `metadata`

| Key                | Type           | Required | Description                                              |
| ------------------ | -------------- | -------- | -------------------------------------------------------- |
| `solver`           | string         | yes      | Pipeline id. Default `"sqpack.feasibility"`.             |
| `timestamp`        | string         | yes      | ISO-8601 UTC, e.g. `"2026-05-26T10:00:00Z"`.             |
| `solver_version`   | string         | no       | sqpack package version at write time.                    |
| `elapsed_seconds`  | float          | no       | Total wall time across all stages.                       |
| `known_best`       | float or null  | no       | Value of `sqpack.known_best.KNOWN_BEST[n]` or null.      |
| `gap_pct`          | float or null  | no       | `100 * (s - known_best) / known_best` or null.           |
| `n_total_trials`   | int            | no       | Total feasibility trials executed.                       |
| `seed`             | int or null    | no       | Top-level seed if any.                                   |
| `args`             | object         | no       | Argparse namespace as a dict.                            |

### `history[i]`

Each entry records one stage transition.

```json
{ "stage": "solve",  "from_s": null,  "to_s": 3.88,  "elapsed_seconds": 3000.0 }
{ "stage": "refine", "from_s": 3.88,  "to_s": 3.878, "elapsed_seconds": 500.0 }
{ "stage": "polish", "from_s": 3.878, "to_s": 3.877, "elapsed_seconds": 100.0 }
```

`stage` is one of `solve`, `refine`, `polish`. `from_s` may be `null` for
the initial entry. `elapsed_seconds` may be `null` if not recorded.

### `top_10[i]`

```json
{ "s": 3.877, "squares": [ {"x": ..., "y": ..., "theta": ...}, ... ] }
```

Same `squares` shape as the top level. Sorted ascending by `s`.

## Worked example

```jsonc
{
  "schema_version": "1.0",
  "n": 11,
  "s": 3.877083123,
  "squares": [
    { "x": 0.5, "y": 0.5, "theta": 0.0 }
  ],
  "metadata": {
    "solver": "sqpack.feasibility",
    "solver_version": "0.1.0",
    "timestamp": "2026-05-26T10:00:00Z",
    "elapsed_seconds": 3600.5,
    "known_best": 3.87708359002281,
    "gap_pct": 0.000103,
    "n_total_trials": 1234567,
    "seed": null,
    "args": { "n": 11, "rotations": 2, "numrotate": [5] }
  },
  "history": [
    { "stage": "solve",  "from_s": null,  "to_s": 3.88,  "elapsed_seconds": 3000.0 },
    { "stage": "refine", "from_s": 3.88,  "to_s": 3.878, "elapsed_seconds": 500.0 },
    { "stage": "polish", "from_s": 3.878, "to_s": 3.877083123, "elapsed_seconds": 100.0 }
  ],
  "top_10": []
}
```

## Legacy migration

Pre-v1 files are flat (no `schema_version` key). On `load_result`, sqpack
maps them to v1 in memory using these rules:

| Legacy key       | v1 destination                           |
| ---------------- | ---------------------------------------- |
| `known_best`     | `metadata.known_best`                    |
| `gap_pct`        | `metadata.gap_pct`                       |
| `n_total`        | `metadata.n_total_trials`                |
| `elapsed`        | `metadata.elapsed_seconds`               |
| `timestamp`      | `metadata.timestamp`                     |
| `args`           | `metadata.args`                          |
| `optimized_from` | one synthesized `history` entry (refine) |
| `polished_from`  | one synthesized `history` entry (polish) |
| `top_10`         | preserved unchanged                      |
| `squares`        | preserved unchanged                      |

The on-disk file is not modified by `load_result`; call `save_result` to
persist the v1 form.
