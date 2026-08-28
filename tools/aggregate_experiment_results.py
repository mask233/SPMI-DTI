#!/usr/bin/env python3
"""Aggregate seed metrics and compare methods on paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paired_stats import mean, paired_summary, sample_sd  # noqa: E402


METRICS = ["AUC", "PRC", "Accuracy", "Recall", "Precision"]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("provide a non-empty, duplicate-free integer list")
    return values


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be NAME=RESULTS_ROOT")
    name, root = value.split("=", 1)
    name = name.strip()
    if not name or not root.strip():
        raise argparse.ArgumentTypeError("method must be NAME=RESULTS_ROOT")
    return name, Path(root.strip())


def finite_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: finite_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json_value(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values, preserving None entries."""
    indexed = sorted(
        ((index, value) for index, value in enumerate(p_values) if value is not None),
        key=lambda item: item[1],
    )
    adjusted: list[float | None] = [None] * len(p_values)
    running_max = 0.0
    count = len(indexed)
    for rank, (original_index, value) in enumerate(indexed):
        candidate = min(1.0, (count - rank) * value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--method", action="append", type=parse_method, required=True,
                        help="repeatable NAME=RESULTS_ROOT; first method is the default reference")
    parser.add_argument("--reference", help="method name used as paired-comparison numerator")
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("1,2,3,4,5"))
    parser.add_argument("--output-dir", default="analysis/formal_results")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    methods = dict(args.method)
    if len(methods) != len(args.method):
        raise SystemExit("duplicate --method name")
    reference = args.reference or next(iter(methods))
    if reference not in methods:
        raise SystemExit(f"--reference {reference!r} is not one of: {', '.join(methods)}")
    output_dir = Path(args.output_dir)
    output_files = [
        output_dir / "formal_results.json",
        output_dir / "metric_summary.csv",
        output_dir / "paired_comparisons.csv",
        output_dir / "formal_results.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.force:
        raise SystemExit("refusing to replace reports; use --force: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    values: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for dataset in args.datasets:
        values[dataset] = {}
        for method, root in methods.items():
            values[dataset][method] = {metric: {} for metric in METRICS}
            for seed in args.seeds:
                path = root / dataset / f"seed_{seed}" / "outer_test_metrics.json"
                if not path.is_file():
                    raise SystemExit(f"missing metrics: {path}")
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if record.get("dataset") != dataset or record.get("seed") != seed:
                    raise SystemExit(f"dataset/seed metadata mismatch: {path}")
                for metric in METRICS:
                    try:
                        metric_value = float(record["metrics"][metric])
                    except (KeyError, TypeError, ValueError):
                        raise SystemExit(f"missing or invalid {metric}: {path}") from None
                    if not math.isfinite(metric_value):
                        raise SystemExit(f"non-finite {metric}: {path}")
                    values[dataset][method][metric][seed] = metric_value

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for method in methods:
            for metric in METRICS:
                ordered = [values[dataset][method][metric][seed] for seed in args.seeds]
                summary_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "n": len(ordered),
                    "mean": mean(ordered),
                    "sample_sd": sample_sd(ordered),
                    "seeds": ",".join(map(str, args.seeds)),
                    "values": ",".join(f"{value:.12g}" for value in ordered),
                })
        for comparator in methods:
            if comparator == reference:
                continue
            for metric in METRICS:
                reference_values = [values[dataset][reference][metric][seed] for seed in args.seeds]
                comparator_values = [values[dataset][comparator][metric][seed] for seed in args.seeds]
                statistics = paired_summary(reference_values, comparator_values)
                paired_rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "reference": reference,
                    "comparator": comparator,
                    "difference_definition": f"{reference} - {comparator}",
                    **statistics,
                })

    for raw_field, adjusted_field in (
        ("paired_t_p", "paired_t_p_holm"),
        ("wilcoxon_exact_p", "wilcoxon_exact_p_holm"),
    ):
        adjusted_values = holm_adjust([row[raw_field] for row in paired_rows])
        for row, adjusted in zip(paired_rows, adjusted_values):
            row[adjusted_field] = adjusted

    write_csv(
        output_files[1], summary_rows,
        ["dataset", "method", "metric", "n", "mean", "sample_sd", "seeds", "values"],
    )
    paired_fields = [
        "dataset", "metric", "reference", "comparator", "difference_definition",
        "n_pairs", "mean_difference", "sample_sd_difference", "ci95_low", "ci95_high",
        "paired_t", "paired_t_p", "paired_t_p_holm", "cohen_dz",
        "wilcoxon_exact_p", "wilcoxon_exact_p_holm",
    ]
    write_csv(output_files[2], paired_rows, paired_fields)

    report = finite_json_value({
        "status": "ok",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "methods": {name: str(root) for name, root in methods.items()},
        "reference": reference,
        "metric_summary": summary_rows,
        "paired_comparisons": paired_rows,
        "notes": [
            "sample_sd uses ddof=1 across independent seeds",
            "paired differences are reference minus comparator on identical seeds/splits",
            "95% CI is the Student-t interval for the mean paired difference",
            "exact two-sided Wilcoxon signed-rank p-values omit zero differences",
            "Holm-Bonferroni adjusted p-values control family-wise error across this report",
        ],
    })
    output_files[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = [
        "# Formal experiment summary",
        "",
        f"Seeds: `{','.join(map(str, args.seeds))}`. Values are mean ± sample SD.",
        "",
        "| Dataset | Method | Metric | Mean ± SD |",
        "|---|---|---|---:|",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['dataset']} | {row['method']} | {row['metric']} | "
            f"{row['mean']:.4f} ± {row['sample_sd']:.4f} |"
        )
    if paired_rows:
        markdown.extend([
            "",
            "## Paired comparisons",
            "",
            "Differences are reference minus comparator.",
            "",
            "| Dataset | Metric | Comparison | Δ mean (95% CI) | paired-t p (Holm) | Wilcoxon p (Holm) |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in paired_rows:
            wilcoxon = row["wilcoxon_exact_p"]
            wilcoxon_text = "NA" if wilcoxon is None else f"{wilcoxon:.4g}"
            wilcoxon_holm = row["wilcoxon_exact_p_holm"]
            wilcoxon_holm_text = "NA" if wilcoxon_holm is None else f"{wilcoxon_holm:.4g}"
            markdown.append(
                f"| {row['dataset']} | {row['metric']} | {row['reference']} vs {row['comparator']} | "
                f"{row['mean_difference']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['paired_t_p']:.4g} ({row['paired_t_p_holm']:.4g}) | "
                f"{wilcoxon_text} ({wilcoxon_holm_text}) |"
            )
    output_files[3].write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "reference": reference,
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
