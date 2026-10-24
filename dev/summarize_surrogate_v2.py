#!/usr/bin/env python3
"""Conditional paired-replicate diagnostics, never ratio confidence intervals."""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def summarize(root):
    root = Path(root)
    result = {"scope": "conditional MC replicate errors; calibration uncertainty omitted; ratios descriptive"}
    for stem in ("heldout_flux", "historical_flux", "heldout_derivatives", "historical_derivatives"):
        path = root / (stem + ".csv")
        if not path.exists():
            continue
        groups = defaultdict(list)
        with path.open() as f:
            for r in csv.DictReader(f):
                key = tuple(r.get(k, "") for k in ("bin", "field", "cell", "fraction"))
                groups[key].append({k: float(v) for k, v in r.items()})
        rows = []
        for key, reps in groups.items():
            truth = "mc_secant" if "derivatives" in stem else "mc"
            mean_mc = st.mean(r[truth] for r in reps)
            row = {"key": key, "replicates": len(reps), "mc_mean": mean_mc}
            for mode in ("before", "after"):
                diffs = [r[mode] - r[truth] for r in reps]
                se = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
                row[mode + "_bias"] = st.mean(diffs)
                row[mode + "_se"] = se
                row[mode + "_relative_bias"] = st.mean(diffs) / max(abs(mean_mc), 1e-30)
                row[mode + "_over_2se"] = abs(st.mean(diffs)) > 2 * se
            # Paired before/after improvement dispersion retains correlation.
            gains = [abs(r["before"] - r[truth]) - abs(r["after"] - r[truth]) for r in reps]
            row["paired_absolute_error_gain_mean"] = st.mean(gains)
            row["paired_absolute_error_gain_se"] = st.stdev(gains) / math.sqrt(len(gains))
            if "ad" in reps[0]:
                row["ad_fd_relative_error"] = max(abs(r["ad"]-r["fd"]) / max(abs(r["ad"]), abs(r["fd"]), 1e-30) for r in reps)
            if "column_water_fraction" in reps[0]:
                row["column_water_fraction"] = reps[0]["column_water_fraction"]
            rows.append(row)
        metrics = {"groups": len(rows), "rows": rows}
        for mode in ("before", "after"):
            biases = [abs(r[mode + "_relative_bias"]) for r in rows]
            metrics[mode + "_median_absolute_relative_bias"] = st.median(biases)
            metrics[mode + "_max_absolute_relative_bias"] = max(biases)
            metrics[mode + "_count_over_2se"] = sum(r[mode + "_over_2se"] for r in rows)
        metrics["median_absolute_relative_bias_improvement_fraction"] = 1 - metrics["after_median_absolute_relative_bias"] / max(metrics["before_median_absolute_relative_bias"], 1e-30)
        if "flux" in stem:
            metrics["prespecified_flux_gate_pass"] = metrics["median_absolute_relative_bias_improvement_fraction"] >= .3 and metrics["after_max_absolute_relative_bias"] <= metrics["before_max_absolute_relative_bias"]
        else:
            metrics["derivative_median_and_count_nonworsening"] = metrics["after_median_absolute_relative_bias"] <= metrics["before_median_absolute_relative_bias"] and metrics["after_count_over_2se"] <= metrics["before_count_over_2se"]
            if "ad" in next(iter(groups.values()))[0]:
                metrics["max_ad_fd_relative_error"] = max(r["ad_fd_relative_error"] for r in rows)
        fields = sorted(set(r["key"][1] for r in rows) - {""})
        metrics["by_field"] = {}
        for field in fields:
            subset = [r for r in rows if r["key"][1] == field]
            before = st.median(abs(r["before_relative_bias"]) for r in subset)
            after = st.median(abs(r["after_relative_bias"]) for r in subset)
            entry = dict(groups=len(subset), before_median_absolute_relative_bias=before,
                         after_median_absolute_relative_bias=after,
                         improvement_fraction=1-after/max(before, 1e-30),
                         before_count_over_2se=sum(r["before_over_2se"] for r in subset),
                         after_count_over_2se=sum(r["after_over_2se"] for r in subset))
            if "column_water_fraction" in subset[0]:
                entry.update(q_min=min(r["column_water_fraction"] for r in subset),
                             q_max=max(r["column_water_fraction"] for r in subset))
            metrics["by_field"][field] = entry
        result[stem] = metrics
    (root / "paired_assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"} if isinstance(v, dict) else v for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    summarize(sys.argv[1])
