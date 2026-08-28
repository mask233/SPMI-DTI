#!/usr/bin/env python3
"""Generate frozen nested random-split manifests without starting training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.DataPrepare import get_kfold_data  # noqa: E402
from utils.split_manifest import (  # noqa: E402
    file_sha256,
    load_dti_rows,
    model_pair_sha256,
    parse_dti_row,
    save_split_manifest,
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one integer")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return values


def build_splits(rows: list[str], seed: int, folds: int) -> tuple[list[str], list[str], list[tuple[list[str], list[str]]]]:
    shuffled = list(rows)
    # This is the isolated equivalent of np.random.seed(seed) followed by
    # np.random.shuffle(), which is what the training code currently uses.
    np.random.RandomState(seed).shuffle(shuffled)
    split_pos = len(shuffled) - int(len(shuffled) * 0.2)
    trainval = shuffled[:split_pos]
    outer_test = shuffled[split_pos:]
    inner = [get_kfold_data(index, trainval, k=folds) for index in range(folds)]
    return trainval, outer_test, inner


def validate_model_pairs(rows: list[str], dataset: str) -> None:
    labels_by_pair: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        _, _, smiles, sequence, label = parse_dti_row(row)
        pair_id = model_pair_sha256(smiles, sequence)
        counts[pair_id] += 1
        labels_by_pair.setdefault(pair_id, set()).add(label)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    conflicts = sum(1 for labels in labels_by_pair.values() if len(labels) > 1)
    if duplicates or conflicts:
        raise ValueError(
            f"{dataset}: formal split generation requires unique model-input pairs; "
            f"duplicate_rows={duplicates}, conflicting_groups={conflicts}"
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset basenames, e.g. Davis DrugBank BIOSNAP")
    parser.add_argument("--data-dir", default=os.environ.get("DTI_DATA_DIR", "DataSets"))
    parser.add_argument("--output-root", default="experiment_splits")
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("1,2,3,4,5"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="replace only the generated manifest/report files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    protocol_path = output_root / "split_protocol.json"
    planned_files = [protocol_path]
    for dataset in args.datasets:
        for seed in args.seeds:
            seed_dir = output_root / dataset / f"seed_{seed}"
            planned_files.append(seed_dir / "outer_split_manifest.csv")
            planned_files.extend(seed_dir / str(fold) / "split_manifest.csv" for fold in range(1, args.folds + 1))
    existing = [str(path) for path in planned_files if path.exists()]
    if existing and not args.force:
        preview = "\n  ".join(existing[:10])
        raise SystemExit(f"Refusing to replace {len(existing)} existing generated file(s); use --force:\n  {preview}")

    # Validate every requested dataset before creating any manifest.  A stale
    # dataset later in the argument list must not leave a partially frozen
    # multi-dataset protocol behind.
    validated: dict[str, tuple[Path, list[str], Counter[str]]] = {}
    for dataset in args.datasets:
        dataset_path = data_dir / f"{dataset}.txt"
        try:
            rows = load_dti_rows(dataset_path)
            validate_model_pairs(rows, dataset)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        label_counts = Counter(parse_dti_row(row)[-1] for row in rows)
        validated[dataset] = (dataset_path, rows, label_counts)

    protocol: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "legacy NumPy RandomState shuffle; outer 80/20 row split; K contiguous inner folds",
        "seeds": args.seeds,
        "inner_folds": args.folds,
        "datasets": {},
    }
    for dataset in args.datasets:
        dataset_path, rows, label_counts = validated[dataset]
        dataset_record: dict[str, object] = {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "rows": len(rows),
            "labels": dict(sorted(label_counts.items())),
            "seeds": {},
        }
        for seed in args.seeds:
            trainval, outer_test, inner = build_splits(rows, seed=seed, folds=args.folds)
            seed_dir = output_root / dataset / f"seed_{seed}"
            save_split_manifest(
                seed_dir / "outer_split_manifest.csv",
                {"trainval": trainval, "outer_test": outer_test},
                seed=seed,
                fold=None,
            )
            fold_counts: list[dict[str, int]] = []
            for fold_index, (train_rows, valid_rows) in enumerate(inner, start=1):
                save_split_manifest(
                    seed_dir / str(fold_index) / "split_manifest.csv",
                    {"train": train_rows, "valid": valid_rows, "outer_test": outer_test},
                    seed=seed,
                    fold=fold_index,
                )
                fold_counts.append({"fold": fold_index, "train": len(train_rows), "valid": len(valid_rows)})
            dataset_record["seeds"][str(seed)] = {
                "trainval": len(trainval),
                "outer_test": len(outer_test),
                "folds": fold_counts,
            }
        protocol["datasets"][dataset] = dataset_record
    write_json(protocol_path, protocol)
    print(json.dumps({
        "status": "ok",
        "output_root": str(output_root),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "inner_folds": args.folds,
        "protocol": str(protocol_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
