# sqpack CLI reference

Installing the package puts a single `sqpack` script on `PATH`. Subcommands:

```
sqpack solve N [flags]
sqpack refine RESULT.json [flags]
sqpack render RESULT.json [flags]
sqpack info RESULT.json
```

Every command exits 0 on success, non-zero on argument errors.

---

## `sqpack solve N`

Run the full pipeline (feasibility search, refine, polish) for `n = N`
unit squares. On every improvement to the best `s` seen so far, writes
the current best to a JSON file under `--output-dir`.

| Flag                            | Default     | Description |
| ------------------------------- | ----------- | ----------- |
| `N`                             | required    | Number of unit squares. |
| `--steps`                       | `1000`      | Max compression steps per feasibility trial. |
| `--compress-restarts`           | `1`         | Compression restarts per drop. |
| `--rotations`                   | `2`         | Distinct rotational angles per trial. |
| `--numrotate`                   | required when `--rotations > 1` | Comma-separated count of squares per non-zero rotation, e.g. `"5"` or `"1,4"`. Must have `rotations - 1` values and sum `<= N`. |
| `--angle-power`                 | `8`         | Angle sampling bias power. `1` = uniform. |
| `--s-start`                     | `ceil(sqrt(N))` | Initial container side. |
| `--cohort-size`                 | `10`        | Feasibles to collect per cohort before refining the best. |
| `--time`                        | none        | Time budget in seconds. |
| `--workers`, `-w`               | `1`         | Parallel workers. `0` auto-detects (cpu_count). |
| `--refine-slide-rounds`         | `50`        | SA refinement rounds per top-10 breach. |
| `--refine-slide-sa-steps`       | `500000`    | SA steps per refinement round. |
| `--refine-sa-t-start`           | `0.5`       | SA coarse temperature. |
| `--refine-sa-t-end`             | `1e-7`      | SA fine temperature. |
| `--refine-bisect-steps`         | `6`         | Angle bisection steps. |
| `--refine-bisect-slide-steps`   | `2000`      | Slide steps per bisection candidate. |
| `--refine-bisect-slide-rounds`  | `20`        | Slide rounds per bisection candidate. |
| `--refine-explode-factor`       | `1.05`      | Explode factor for angle bisection. |
| `--no-polish`                   | off         | Disable the theta polish stage. |
| `--polish-theta-range`          | `1.0`       | Polish theta scan range (degrees). |
| `--polish-theta-points`         | `11`        | Polish theta scan points per iteration. |
| `--polish-explode`              | `1.05`      | Polish explode factor. |
| `--polish-sa-rounds`            | `20`        | Polish SA rounds per theta. |
| `--polish-sa-steps`             | `200000`    | Polish SA steps per round. |
| `--polish-convergence-tol`      | `0.01`      | Polish convergence tolerance (degrees). |
| `--seed`                        | none        | Top-level RNG seed. |
| `--save`                        | none        | If set, save a PNG of the final packing here. |
| `--output-dir`                  | `output`    | Directory for JSON result files. |
| `-v`, `--verbose`               | off         | Verbose progress. |

### Example: long search for n = 26

```
sqpack solve 26 \
    --rotations 2 --numrotate 8 \
    --cohort-size 20 \
    --time 86400 \
    --workers 0 \
    --output-dir output/n26_sweep \
    -v
```

---

## `sqpack refine RESULT.json`

Apply iterative refinement (multi-axis sweep, coordinate descent, SA,
global shrink) to an existing packing. Reads legacy and v1 JSON; writes v1.

| Flag             | Default      | Description |
| ---------------- | ------------ | ----------- |
| `RESULT.json`    | required     | Input JSON. |
| `--rounds`       | `100`        | Optimization rounds. |
| `--sa-steps`     | `100000`     | SA steps per round. |
| `-o`, `--output` | input path   | Output JSON path. Defaults to overwriting the input. |
| `--save`         | none         | Save PNG of the refined packing here. |
| `-v`, `--verbose`| off          | Verbose progress. |

---

## `sqpack render RESULT.json`

Render a packing JSON to a PNG.

| Flag             | Default            | Description |
| ---------------- | ------------------ | ----------- |
| `RESULT.json`    | required           | Input JSON. |
| `-o`, `--output` | `<input>.png`      | Output PNG path. |
| `--show`         | off                | Display the figure interactively. |
| `--rank K`       | none               | Render the K-th best from `top_10` (1-indexed) instead of the top-level packing. |

---

## `sqpack info RESULT.json`

Print a one-screen summary of a result file: schema version, n, s, gap,
solver/version, timestamps, history stages, and the top_10 list.
