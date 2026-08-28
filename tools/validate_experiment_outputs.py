#!/usr/bin/env python3
"""Validate formal-run manifests, checkpoints, predictions, and metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.binary_metrics import binary_metrics  # noqa: E402
from utils.split_manifest import file_sha256  # noqa: E402


METRICS = ["AUC", "PRC", "Accuracy", "Recall", "Precision"]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("provide a non-empty, duplicate-free integer list")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_manifests(actual: Path, frozen: Path, errors: list[str]) -> list[dict[str, str]]:
    if not frozen.is_file():
        errors.append(f"missing frozen manifest: {frozen}")
        return []
    expected_rows = read_csv(frozen)
    if not actual.is_file():
        errors.append(f"missing run manifest: {actual}")
        return expected_rows
    actual_rows = read_csv(actual)
    if actual_rows != expected_rows:
        errors.append(f"run manifest differs from frozen manifest: {actual}")
    return expected_rows


def validate_prediction_csv(
    path: Path,
    expected_rows: list[dict[str, str]],
    seed: int,
    fold_label: str,
    errors: list[str],
) -> tuple[list[int], list[int], list[float]] | None:
    if not path.is_file():
        errors.append(f"missing prediction file: {path}")
        return None
    rows = read_csv(path)
    if len(rows) != len(expected_rows):
        errors.append(f"prediction count {len(rows)} != {len(expected_rows)}: {path}")
        return None
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for index, (row, expected) in enumerate(zip(rows, expected_rows)):
        location = f"{path}:row {index + 2}"
        try:
            truth = int(row["y_true"])
            prediction = int(row["y_pred"])
            score = float(row["y_score"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid prediction fields at {location}: {exc}")
            return None
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            errors.append(f"non-probability y_score={score!r} at {location}")
        expected_prediction = int(
            score >= 0.5 if fold_label == "probability_mean_over_inner_folds" else score > 0.5
        )
        if prediction != expected_prediction:
            errors.append(f"y_pred is inconsistent with the 0.5 threshold at {location}")
        expected_identity = (expected["sample_index"], expected["sample_id"], expected["label"])
        actual_identity = (row.get("sample_index"), row.get("sample_id"), row.get("y_true"))
        if actual_identity != expected_identity:
            errors.append(f"prediction/sample mismatch at {location}")
        if row.get("seed") != str(seed):
            errors.append(f"wrong seed at {location}: {row.get('seed')!r}")
        if row.get("fold") != fold_label:
            errors.append(f"wrong fold label at {location}: {row.get('fold')!r}")
        if row.get("split") != "outer_test":
            errors.append(f"wrong split at {location}: {row.get('split')!r}")
        y_true.append(truth)
        y_pred.append(prediction)
        y_score.append(score)
        if len(errors) >= 100:
            errors.append("stopped detailed validation after 100 errors")
            return None
    if len({row["sample_id"] for row in rows}) != len(rows):
        errors.append(f"duplicate sample_id values in predictions: {path}")
    return y_true, y_pred, y_score


def close_enough(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("1,2,3,4,5"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--data-dir", default=os.environ.get("DTI_DATA_DIR", "DataSets"))
    parser.add_argument("--results-root", default=".")
    parser.add_argument("--frozen-splits", default="experiment_splits")
    parser.add_argument("--skip-checkpoints", action="store_true", help="for synced copies that omit *.pth")
    parser.add_argument("--output", help="optional JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    frozen_root = Path(args.frozen_splits)
    data_dir = Path(args.data_dir)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    protocol_path = frozen_root / "split_protocol.json"
    if not protocol_path.is_file():
        errors.append(f"missing split protocol: {protocol_path}")
        protocol: dict[str, Any] = {}
    else:
        with protocol_path.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        if protocol.get("inner_folds") != args.folds:
            errors.append("--folds differs from frozen split protocol")

    for dataset in args.datasets:
        dataset_errors_before = len(errors)
        protocol_dataset = protocol.get("datasets", {}).get(dataset)
        dataset_path = data_dir / f"{dataset}.txt"
        if protocol_dataset is None:
            errors.append(f"dataset absent from frozen protocol: {dataset}")
        elif not dataset_path.is_file():
            errors.append(f"missing dataset: {dataset_path}")
        else:
            current_hash = file_sha256(dataset_path)
            if current_hash != protocol_dataset.get("sha256"):
                errors.append(f"dataset SHA-256 differs from frozen protocol: {dataset}")
        dataset_detail: dict[str, Any] = {"seeds": {}}
        for seed in args.seeds:
            seed_dir = results_root / dataset / f"seed_{seed}"
            frozen_seed_dir = frozen_root / dataset / f"seed_{seed}"
            seed_errors_before = len(errors)
            outer_rows = compare_manifests(
                seed_dir / "outer_split_manifest.csv",
                frozen_seed_dir / "outer_split_manifest.csv",
                errors,
            )
            expected_outer = [row for row in outer_rows if row.get("split") == "outer_test"]
            fold_prediction_values: list[tuple[list[int], list[int], list[float]]] = []
            for fold in range(1, args.folds + 1):
                run_fold_dir = seed_dir / str(fold)
                compare_manifests(
                    run_fold_dir / "split_manifest.csv",
                    frozen_seed_dir / str(fold) / "split_manifest.csv",
                    errors,
                )
                if not args.skip_checkpoints:
                    checkpoints = list(run_fold_dir.glob("valid_best_checkpoint-*.pth"))
                    if len(checkpoints) != 1:
                        errors.append(f"expected exactly one checkpoint in {run_fold_dir}, found {len(checkpoints)}")
                fold_prediction = run_fold_dir / f"{dataset}_outer_test_predictions_seed_{seed}_fold_{fold}.csv"
                fold_values = validate_prediction_csv(
                    fold_prediction, expected_outer, seed, str(fold), errors
                )
                if fold_values is not None:
                    fold_prediction_values.append(fold_values)
                for optional_name in (
                    f"{dataset}_train_predictions_seed_{seed}_fold_{fold}.csv",
                    f"{dataset}_valid_predictions_seed_{seed}_fold_{fold}.csv",
                    "The_results_of_whole_dataset.txt",
                ):
                    if not (run_fold_dir / optional_name).is_file():
                        warnings.append(f"missing diagnostic output: {run_fold_dir / optional_name}")

            ensemble_path = seed_dir / f"{dataset}_outer_test_ensemble_predictions_seed_{seed}.csv"
            prediction_values = validate_prediction_csv(
                ensemble_path,
                expected_outer,
                seed,
                "probability_mean_over_inner_folds",
                errors,
            )
            if prediction_values is not None and len(fold_prediction_values) == args.folds:
                ensemble_scores = prediction_values[2]
                for index, ensemble_score in enumerate(ensemble_scores):
                    expected_score = sum(values[2][index] for values in fold_prediction_values) / args.folds
                    if not close_enough(ensemble_score, expected_score):
                        errors.append(
                            f"ensemble score is not the mean of {args.folds} fold scores at "
                            f"{ensemble_path}:row {index + 2}; saved={ensemble_score}, "
                            f"expected={expected_score}"
                        )
                        if len(errors) >= 100:
                            errors.append("stopped ensemble-mean validation after 100 errors")
                            break
            metrics_path = seed_dir / "outer_test_metrics.json"
            recomputed: dict[str, float] | None = None
            if not metrics_path.is_file():
                errors.append(f"missing seed metrics: {metrics_path}")
            elif prediction_values is not None:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                if stored.get("dataset") != dataset or stored.get("seed") != seed:
                    errors.append(f"dataset/seed metadata mismatch: {metrics_path}")
                if stored.get("inner_folds") != args.folds:
                    errors.append(f"inner_folds mismatch: {metrics_path}")
                if stored.get("n_outer_test") != len(expected_outer):
                    errors.append(f"n_outer_test mismatch: {metrics_path}")
                recomputed = binary_metrics(*prediction_values)
                for metric in METRICS:
                    try:
                        saved_value = float(stored["metrics"][metric])
                    except (KeyError, TypeError, ValueError):
                        errors.append(f"missing/invalid {metric}: {metrics_path}")
                        continue
                    if not close_enough(saved_value, recomputed[metric]):
                        errors.append(
                            f"{metric} mismatch in {metrics_path}: saved={saved_value}, "
                            f"recomputed={recomputed[metric]}"
                        )
            dataset_detail["seeds"][str(seed)] = {
                "outer_test_rows": len(expected_outer),
                "recomputed_metrics": recomputed,
                "status": "ok" if len(errors) == seed_errors_before else "error",
            }
        dataset_detail["status"] = "ok" if len(errors) == dataset_errors_before else "error"
        details[dataset] = dataset_detail

    report = {
        "status": "ok" if not errors else "error",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "inner_folds": args.folds,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
