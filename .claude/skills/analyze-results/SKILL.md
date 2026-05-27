---
name: analyze-results
description: Inspect a sqpack result JSON, compare against KNOWN_BEST, and produce a short report. Use this skill when the user says "inspect this result", "compare to known best", "summarize top_10", "what's the gap on this run", or anything that means "tell me how good this packing is".
---

# Analyze a sqpack result

The user wants a quick read on a result JSON: how does it compare to
the literature value, and what's worth a closer look.

## Steps

1. Run `sqpack info <path>` to print the headline numbers.
   This works on both v1 and legacy JSON (auto-migrated on read).
2. If `gap_pct` is below ~1e-4%, the packing matches the literature
   value. Above that, note the gap explicitly.
3. If `top_10` has entries, list the top 3 and call out any whose `s`
   is within polish noise of the headline `s` — these are alternative
   near-optimal configurations.
4. Offer to:
   - Render a PNG via `sqpack render <path> -o <out.png>`.
   - Render the K-th top-10 entry via `sqpack render <path> --rank K`.
   - Re-refine with `sqpack refine <path> --rounds 200 -v` if the gap
     is non-trivial (>1e-3%).

## Useful output to extract

| From `sqpack info`                | Use it for                            |
| --------------------------------- | ------------------------------------- |
| `n`, `s`, `gap_pct`               | Headline summary.                     |
| `schema_version`                  | If absent or empty, the input is legacy. |
| `metadata.timestamp`              | When this run was produced.           |
| `metadata.elapsed_seconds`        | Total wall time spent on the run.     |
| `history`                         | Which stages ran (solve/refine/polish). |
| `top_10`                          | Diversity of near-best configurations. |

## Report shape

Keep the response short:

```
n=N  s=S.SSSSSSSSSS  gap=G.GGGG%   (matches/beats/below known best)
ran at <timestamp>, elapsed <T>s
history: solve -> ... -> polish
top_10: K entries; top 3 within ~Δ of best
```

Recommend a next step only if the user asked for one or if the result
looks degraded (large gap, validation issue, stale schema).
