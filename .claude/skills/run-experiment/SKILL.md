---
name: run-experiment
description: Launch a sqpack solve for a given n with sensible defaults and watch the output JSON for gap-vs-known-best. Use this skill when the user says "run a solve for n=X", "try to beat n=X", "kick off an n=X sweep", or anything that means "start the solver against n unit squares".
---

# Run a sqpack experiment

The user wants to start a `sqpack solve` run for some `n` and let it
hunt for an improved packing.

## How to invoke

```
sqpack solve N --rotations R --numrotate K --cohort-size C --time T \
    --workers 0 --output-dir output/n${N}_run -v
```

Defaults:

| Param            | Default          |
| ---------------- | ---------------- |
| `--rotations`    | `2`              |
| `--numrotate`    | depends on `n` (ask if unclear) |
| `--cohort-size`  | `10` (raise to 20 for n > 20) |
| `--time`         | `3600` (1 hour). Confirm with user for long runs. |
| `--workers`      | `0` (auto-detect) |
| `--output-dir`   | `output/n${N}_run` |

## Steps

1. Confirm `n` and the `--numrotate` value with the user. For
   unfamiliar `n`, ask before guessing.
2. Make sure the output directory exists and is writable.
3. Kick off the solver in the background (long-running):
   `sqpack solve N ... > /tmp/sqpack_n${N}.log 2>&1 &`
4. Tail the log periodically. Report status every ~minute:
   "trials so far, feasible count, best s, gap vs `KNOWN_BEST[n]`".
5. When the run completes (or the user stops it), summarize:
   - Final `s` and gap percentage.
   - Whether it matches or beats the known best.
   - Path to the result JSON.

## Useful commands

```
sqpack info output/n${N}_run/sqpack_n_${N}_*.json
sqpack render output/n${N}_run/sqpack_n_${N}_*.json -o /tmp/n${N}.png
```

## Caveats

- Long runs (`--time > 3600`) should be launched under `screen` or
  `nohup`, not as a fire-and-forget background job in the current
  shell, unless the user has already set that up.
- The first run after a fresh install pays Numba's JIT compile cost
  (~10s). Don't report a stuck solver before 30 seconds have elapsed.
