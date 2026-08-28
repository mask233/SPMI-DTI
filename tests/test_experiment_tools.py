from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.aggregate_experiment_results import main as aggregate_main
from tools.generate_split_manifests import main as generate_main
from tools.validate_experiment_outputs import main as validate_main
from utils.binary_metrics import binary_metrics
from utils.paired_stats import exact_wilcoxon_signed_rank_p, paired_t_two_sided_p, t_critical_975


PREDICTION_FIELDS = [
    "sample_index", "sample_id", "y_true", "y_score", "y_pred", "seed", "fold", "split"
]


class ExperimentToolTests(unittest.TestCase):
    def test_dependency_free_math(self):
        self.assertAlmostEqual(t_critical_975(4), 2.776445105, places=8)
        self.assertAlmostEqual(paired_t_two_sided_p(t_critical_975(4), 4), 0.05, places=10)
        self.assertEqual(exact_wilcoxon_signed_rank_p([1, 2, 3, 4, 5]), 0.0625)
        metrics = binary_metrics([0, 1], [0, 1], [0.1, 0.9])
        self.assertTrue(all(value == 1.0 for value in metrics.values()))

    def test_generate_and_validate_complete_synthetic_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "DataSets"
            data_dir.mkdir()
            rows = [
                f"D{i} P{i} C{i} SEQ{i} {i % 2}"
                for i in range(50)
            ]
            (data_dir / "Synthetic.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            frozen = root / "splits"
            self.assertEqual(generate_main([
                "Synthetic", "--data-dir", str(data_dir), "--output-root", str(frozen),
                "--seeds", "1", "--folds", "2",
            ]), 0)

            result_root = root / "results"
            seed_dir = result_root / "Synthetic" / "seed_1"
            seed_dir.mkdir(parents=True)
            shutil.copy2(
                frozen / "Synthetic" / "seed_1" / "outer_split_manifest.csv",
                seed_dir / "outer_split_manifest.csv",
            )
            with (seed_dir / "outer_split_manifest.csv").open(newline="", encoding="utf-8") as handle:
                outer = [row for row in csv.DictReader(handle) if row["split"] == "outer_test"]

            def write_predictions(path: Path, fold: str) -> dict[str, float]:
                truth = [int(row["label"]) for row in outer]
                scores = [0.9 if value else 0.1 for value in truth]
                predictions = [int(score >= 0.5) for score in scores]
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
                    writer.writeheader()
                    for row, score, prediction in zip(outer, scores, predictions):
                        writer.writerow({
                            "sample_index": row["sample_index"],
                            "sample_id": row["sample_id"],
                            "y_true": row["label"],
                            "y_score": score,
                            "y_pred": prediction,
                            "seed": 1,
                            "fold": fold,
                            "split": "outer_test",
                        })
                return binary_metrics(truth, predictions, scores)

            for fold in (1, 2):
                run_fold = seed_dir / str(fold)
                run_fold.mkdir()
                shutil.copy2(
                    frozen / "Synthetic" / "seed_1" / str(fold) / "split_manifest.csv",
                    run_fold / "split_manifest.csv",
                )
                (run_fold / "valid_best_checkpoint-cuda0.pth").write_bytes(b"test")
                write_predictions(
                    run_fold / f"Synthetic_outer_test_predictions_seed_1_fold_{fold}.csv",
                    str(fold),
                )
            metrics = write_predictions(
                seed_dir / "Synthetic_outer_test_ensemble_predictions_seed_1.csv",
                "probability_mean_over_inner_folds",
            )
            (seed_dir / "outer_test_metrics.json").write_text(json.dumps({
                "dataset": "Synthetic",
                "seed": 1,
                "inner_folds": 2,
                "n_outer_test": len(outer),
                "metrics": metrics,
            }), encoding="utf-8")
            validation_report = root / "validation.json"
            self.assertEqual(validate_main([
                "Synthetic", "--seeds", "1", "--folds", "2",
                "--data-dir", str(data_dir), "--results-root", str(result_root),
                "--frozen-splits", str(frozen), "--output", str(validation_report),
            ]), 0)
            self.assertEqual(json.loads(validation_report.read_text(encoding="utf-8"))["status"], "ok")

    def test_aggregate_two_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method, offset in (("SPMI", 0.01), ("BASE", 0.0)):
                for seed in range(1, 6):
                    seed_dir = root / method / "Synthetic" / f"seed_{seed}"
                    seed_dir.mkdir(parents=True)
                    metrics = {name: 0.7 + seed * 0.001 + offset for name in (
                        "AUC", "PRC", "Accuracy", "Recall", "Precision"
                    )}
                    (seed_dir / "outer_test_metrics.json").write_text(json.dumps({
                        "dataset": "Synthetic", "seed": seed, "metrics": metrics,
                    }), encoding="utf-8")
            output = root / "summary"
            self.assertEqual(aggregate_main([
                "Synthetic", "--method", f"SPMI={root / 'SPMI'}",
                "--method", f"BASE={root / 'BASE'}", "--seeds", "1,2,3,4,5",
                "--output-dir", str(output),
            ]), 0)
            report = json.loads((output / "formal_results.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(len(report["metric_summary"]), 10)
            self.assertEqual(len(report["paired_comparisons"]), 5)
            self.assertAlmostEqual(report["paired_comparisons"][0]["mean_difference"], 0.01)
            self.assertIn("paired_t_p_holm", report["paired_comparisons"][0])
            self.assertIn("wilcoxon_exact_p_holm", report["paired_comparisons"][0])


if __name__ == "__main__":
    unittest.main()
