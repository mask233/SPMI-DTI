#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, lightweight audit for a server-side DTI dataset and caches.

The script deliberately does not import the training code and never loads a
model.  It can therefore be copied to a server before the large data and
runtime environment are available.  By default JSON is printed to stdout;
``--output`` is required to write a file.

Example::

    python tools/server_audit.py Davis DrugBank BIOSNAP \
        --data-dir DataSets --preprocess-dir DataSets/Preprocessed \
        --output server_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    """Convert a few common optional-library values to JSON-safe values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if record["exists"]:
        try:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
        except Exception as exc:  # permission/read errors should not abort audit
            record["hash_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _iter_dataset_lines(path: Path) -> Iterable[tuple[int, str, list[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if text:
                yield line_number, text, text.split()


def audit_dataset(path: Path, max_names: int) -> dict[str, Any]:
    """Audit the five-column DTI text format without loading it into memory."""
    result: dict[str, Any] = {
        "file": file_record(path),
        "rows": 0,
        "malformed_rows": 0,
        "malformed_line_examples": [],
        "entity_key_definition": (
            "drug_id unless NO-DRUG-ID, otherwise SMILES; "
            "protein_id unless NO-PROT-ID, otherwise sequence"
        ),
        "model_input_pair_definition": "SMILES + protein sequence",
        "unique_drugs": 0,
        "unique_targets": 0,
        "unique_drug_ids": 0,
        "unique_target_ids": 0,
        "unique_smiles": 0,
        "unique_sequences": 0,
        "labels": {},
        "exact_duplicate_rows": 0,
        "exact_duplicate_examples": [],
        "duplicate_pair_rows": 0,
        "conflicting_pair_count": 0,
        "duplicate_pair_examples": [],
        "conflicting_pair_examples": [],
        "model_input_duplicate_pair_rows": 0,
        "model_input_conflicting_pair_count": 0,
        "model_input_duplicate_pair_examples": [],
        "model_input_conflicting_pair_examples": [],
        "no_drug_id_rows": 0,
        "no_drug_id_unique_smiles": 0,
        "no_prot_id_rows": 0,
        "no_prot_id_unique_sequences": 0,
        "no_prot_id_positive_rows": 0,
        "no_prot_id_target_examples": [],
    }
    if not path.is_file():
        result["error"] = "dataset file not found"
        return result

    drugs: set[str] = set()
    targets: set[str] = set()
    drug_ids: set[str] = set()
    target_ids: set[str] = set()
    smiles_values: set[str] = set()
    sequence_values: set[str] = set()
    no_drug_smiles: set[str] = set()
    no_prot_sequences: set[str] = set()
    pair_first_label: dict[tuple[str, str], str] = {}
    model_pair_first_label: dict[tuple[str, str], str] = {}
    seen_rows: set[tuple[str, ...]] = set()
    conflict_pairs: set[tuple[str, str]] = set()
    model_conflict_pairs: set[tuple[str, str]] = set()
    exact_duplicate_examples: list[dict[str, Any]] = []
    duplicate_examples: list[dict[str, Any]] = []
    conflict_examples: list[dict[str, Any]] = []
    model_duplicate_examples: list[dict[str, Any]] = []
    model_conflict_examples: list[dict[str, Any]] = []
    no_prot_targets: set[str] = set()
    labels: Counter[str] = Counter()

    for line_number, text, fields in _iter_dataset_lines(path):
        result["rows"] += 1
        # Current datasets are drug_id, protein_id, SMILES, sequence, label.
        if len(fields) < 5:
            result["malformed_rows"] += 1
            if len(result["malformed_line_examples"]) < max_names:
                result["malformed_line_examples"].append({"line": line_number, "text": text[:240]})
            continue
        drug_id, target_id, smiles, sequence, label = (
            fields[0], fields[1], fields[-3], fields[-2], fields[-1]
        )
        missing_drug_id = drug_id.upper() == "NO-DRUG-ID"
        missing_target_id = target_id.upper() == "NO-PROT-ID"
        drug = smiles if missing_drug_id else drug_id
        target = sequence if missing_target_id else target_id
        row_key = tuple(fields)
        if row_key in seen_rows:
            result["exact_duplicate_rows"] += 1
            if len(exact_duplicate_examples) < max_names:
                exact_duplicate_examples.append({
                    "line": line_number, "drug_id": drug_id, "target_id": target_id,
                    "smiles_prefix": smiles[:80],
                })
        else:
            seen_rows.add(row_key)
        drugs.add(drug)
        targets.add(target)
        smiles_values.add(smiles)
        sequence_values.add(sequence)
        if not missing_drug_id:
            drug_ids.add(drug_id)
        else:
            result["no_drug_id_rows"] += 1
            no_drug_smiles.add(smiles)
        if not missing_target_id:
            target_ids.add(target_id)
        labels[label] += 1
        if missing_target_id:
            result["no_prot_id_rows"] += 1
            no_prot_sequences.add(sequence)
            no_prot_targets.add(target_id)
            if label == "1":
                result["no_prot_id_positive_rows"] += 1
        pair = (drug, target)
        previous = pair_first_label.get(pair)
        if previous is None:
            pair_first_label[pair] = label
        else:
            result["duplicate_pair_rows"] += 1
            if len(duplicate_examples) < max_names:
                duplicate_examples.append({"line": line_number, "drug": drug, "target": target})
            if previous != label and pair not in conflict_pairs:
                conflict_pairs.add(pair)
                if len(conflict_examples) < max_names:
                    conflict_examples.append({
                        "line": line_number, "drug": drug, "target": target,
                        "first_label": previous, "later_label": label,
                    })

        model_pair = (smiles, sequence)
        model_previous = model_pair_first_label.get(model_pair)
        if model_previous is None:
            model_pair_first_label[model_pair] = label
        else:
            result["model_input_duplicate_pair_rows"] += 1
            if len(model_duplicate_examples) < max_names:
                model_duplicate_examples.append({
                    "line": line_number,
                    "smiles_prefix": smiles[:80],
                    "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                })
            if model_previous != label and model_pair not in model_conflict_pairs:
                model_conflict_pairs.add(model_pair)
                if len(model_conflict_examples) < max_names:
                    model_conflict_examples.append({
                        "line": line_number,
                        "smiles_prefix": smiles[:80],
                        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                        "first_label": model_previous,
                        "later_label": label,
                    })

    result["unique_drugs"] = len(drugs)
    result["unique_targets"] = len(targets)
    result["unique_drug_ids"] = len(drug_ids)
    result["unique_target_ids"] = len(target_ids)
    result["unique_smiles"] = len(smiles_values)
    result["unique_sequences"] = len(sequence_values)
    result["labels"] = dict(sorted(labels.items()))
    result["exact_duplicate_examples"] = exact_duplicate_examples
    result["conflicting_pair_count"] = len(conflict_pairs)
    result["duplicate_pair_examples"] = duplicate_examples
    result["conflicting_pair_examples"] = conflict_examples
    result["model_input_conflicting_pair_count"] = len(model_conflict_pairs)
    result["model_input_duplicate_pair_examples"] = model_duplicate_examples
    result["model_input_conflicting_pair_examples"] = model_conflict_examples
    result["no_drug_id_unique_smiles"] = len(no_drug_smiles)
    result["no_prot_id_unique_sequences"] = len(no_prot_sequences)
    result["no_prot_id_target_examples"] = sorted(no_prot_targets)[:max_names]
    return result


def _safe_torch_load(path: Path) -> Any:
    import torch  # type: ignore

    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # torch before the weights_only argument
        return torch.load(path, **kwargs)


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except Exception:
        return None


def _tensor_values(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return value.detach().cpu().reshape(-1).tolist()
    except Exception:
        try:
            return list(value)
        except Exception:
            return None


def _graph_summary(value: Any) -> dict[str, Any]:
    """Summarize one cached Data/Batch without importing project modules."""
    summary: dict[str, Any] = {"type": type(value).__name__}
    if value is None:
        summary["status"] = "none"
        return summary
    x = getattr(value, "x", None)
    summary["x_shape"] = _tensor_shape(x)
    summary["x_feature_dim"] = (_tensor_shape(x) or [None, None])[-1]
    pocket = getattr(value, "pocket_number", None)
    pocket_values = _tensor_values(pocket)
    batch = getattr(value, "batch", None)
    batch_values = _tensor_values(batch)
    if batch_values is not None:
        graph_ids = sorted({int(v) for v in batch_values})
        summary["graph_slots"] = len(graph_ids)
    else:
        summary["graph_slots"] = 1
    if pocket_values is not None:
        summary["pocket_number_min"] = min(pocket_values) if pocket_values else None
        summary["pocket_number_max"] = max(pocket_values) if pocket_values else None
        real = sum(1 for item in pocket_values if item != -1)
        summary["real_nodes"] = real
        summary["virtual_nodes"] = len(pocket_values) - real
        # In this cache representation a virtual slot has pocket_number=-1.
        summary["all_virtual"] = bool(pocket_values) and real == 0
        summary["real_graph_slots"] = None
        if batch_values is not None:
            real_graphs = {int(g) for g, p in zip(batch_values, pocket_values) if p != -1}
            summary["real_graph_slots"] = len(real_graphs)
        try:
            x_cpu = x.detach().cpu()
            pocket_cpu = pocket.detach().cpu().reshape(-1)
            zero_rows = x_cpu.abs().sum(dim=1).eq(0)
            summary["zero_feature_real_nodes"] = int(
                (zero_rows & pocket_cpu.ne(-1)).sum().item())
        except Exception:
            summary["zero_feature_real_nodes"] = None
    else:
        summary["all_virtual"] = None
        summary["real_graph_slots"] = None
    return summary


def audit_cache(path: Path, max_names: int, dataset_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"file": file_record(path), "entries": 0, "none_entries": 0,
                              "missing_or_invalid_examples": [], "feature_dimensions": {},
                              "graph_slots": {}, "real_graph_slots": {}, "all_virtual_entries": 0,
                              "zero_feature_real_entries": 0,
                              "valid_entries": 0, "entry_examples": []}
    if not path.is_file():
        result["error"] = "cache file not found"
        return result
    try:
        cache = _safe_torch_load(path)
    except Exception as exc:
        result["load_error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not isinstance(cache, dict):
        result["load_error"] = f"expected dict, got {type(cache).__name__}"
        return result
    result["entries"] = len(cache)
    dimensions: Counter[str] = Counter()
    slots: Counter[str] = Counter()
    real_slots: Counter[str] = Counter()
    usable_cache_keys: set[str] = set()
    for key, value in cache.items():
        if value is None:
            result["none_entries"] += 1
            if len(result["missing_or_invalid_examples"]) < max_names:
                result["missing_or_invalid_examples"].append(str(key))
            continue
        try:
            summary = _graph_summary(value)
        except Exception as exc:
            if len(result["missing_or_invalid_examples"]) < max_names:
                result["missing_or_invalid_examples"].append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        if summary.get("x_shape") is None:
            if len(result["missing_or_invalid_examples"]) < max_names:
                result["missing_or_invalid_examples"].append(f"{key}: missing x")
            continue
        result["valid_entries"] += 1
        usable_cache_keys.add(str(key))
        dimensions[str(summary.get("x_feature_dim"))] += 1
        slots[str(summary.get("graph_slots"))] += 1
        real_slots[str(summary.get("real_graph_slots"))] += 1
        if len(result["entry_examples"]) < max_names:
            result["entry_examples"].append({"key": str(key), "graph_slots": summary.get("graph_slots"),
                                              "real_graph_slots": summary.get("real_graph_slots"),
                                              "x_feature_dim": summary.get("x_feature_dim"),
                                              "all_virtual": summary.get("all_virtual"),
                                              "zero_feature_real_nodes": summary.get("zero_feature_real_nodes")})
        if summary.get("all_virtual"):
            result["all_virtual_entries"] += 1
        if (summary.get("zero_feature_real_nodes") or 0) > 0:
            result["zero_feature_real_entries"] += 1
    result["feature_dimensions"] = dict(sorted(dimensions.items()))
    result["graph_slots"] = dict(sorted(slots.items()))
    result["real_graph_slots"] = dict(sorted(real_slots.items()))
    if dataset_path is not None and dataset_path.is_file():
        all_cache_keys = {str(key) for key in cache}
        rows = key_matched_rows = usable_rows = 0
        protein_records: set[tuple[str, str]] = set()
        key_matched_records: set[tuple[str, str]] = set()
        usable_records: set[tuple[str, str]] = set()
        missing_examples: list[dict[str, str]] = []
        unusable_examples: list[dict[str, str]] = []
        for _, _, fields in _iter_dataset_lines(dataset_path):
            if len(fields) < 5:
                continue
            target, sequence = fields[1], fields[-2]
            record = (target, sequence)
            protein_records.add(record)
            rows += 1
            matched_key = target if target in all_cache_keys else (
                sequence if sequence in all_cache_keys else None)
            if matched_key is None:
                if len(missing_examples) < max_names:
                    missing_examples.append({"target": target, "sequence_prefix": sequence[:24]})
                continue
            key_matched_rows += 1
            key_matched_records.add(record)
            if matched_key in usable_cache_keys:
                usable_rows += 1
                usable_records.add(record)
            elif len(unusable_examples) < max_names:
                unusable_examples.append({"target": target, "matched_key": matched_key[:80]})
        result["dataset_coverage"] = {
            "dataset_rows": rows,
            "key_matched_rows": key_matched_rows,
            "usable_cache_rows": usable_rows,
            "unique_protein_records": len(protein_records),
            "key_matched_protein_records": len(key_matched_records),
            "usable_cache_protein_records": len(usable_records),
            "missing_key_examples": missing_examples,
            "unusable_key_examples": unusable_examples,
        }
    return result


def _version(module_name: str, distribution: str | None = None) -> str | None:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", None) or importlib.metadata.version(distribution or module_name))
    except Exception:
        try:
            return importlib.metadata.version(distribution or module_name)
        except Exception:
            return None


def _dependency_probe(module_name: str, distribution: str | None = None) -> dict[str, Any]:
    """Return an explicit status instead of making a missing package look silent."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(distribution or module_name)
            except Exception:
                version = None
        return {"available": True, "version": str(version) if version is not None else None}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("sklearn", "scikit-learn"),
        "rdkit": _version("rdkit"),
        "torch": _version("torch"),
        "torch_geometric": _version("torch_geometric", "torch-geometric"),
        "networkx": _version("networkx"),
        "scipy": _version("scipy"),
    }
    info["dependency_status"] = {
        name: _dependency_probe(module, distribution)
        for name, module, distribution in [
            ("torch", "torch", None),
            ("torch_geometric", "torch_geometric", "torch-geometric"),
            ("rdkit", "rdkit", None),
            ("numpy", "numpy", None),
            ("pandas", "pandas", None),
            ("scikit_learn", "sklearn", "scikit-learn"),
            ("networkx", "networkx", None),
            ("scipy", "scipy", None),
        ]
    }
    try:
        import torch  # type: ignore

        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            info["gpu_count"] = 0
            info["gpus"] = []
    except Exception as exc:
        info["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return info


def find_cache_files(data_dir: Path, preprocess_dir: Path | None, dataset: str) -> list[Path]:
    candidates: list[Path] = []
    roots = [preprocess_dir] if preprocess_dir is not None else []
    roots.extend([data_dir / "Preprocessed", data_dir / "Preprocessed_old_backup"])
    seen: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        for candidate in (root / f"{dataset}-subpocket.pt", root / f"{dataset}_subpocket.pt"):
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser()
    preprocess_dir = Path(args.preprocess_dir).expanduser() if args.preprocess_dir else None
    datasets: dict[str, Any] = {}
    for dataset in args.datasets:
        dataset_path = data_dir / f"{dataset}.txt"
        caches = find_cache_files(data_dir, preprocess_dir, dataset)
        datasets[dataset] = {
            "dataset": audit_dataset(dataset_path, args.max_list),
            "caches": [
                audit_cache(path, args.max_list, dataset_path=dataset_path)
                for path in caches if path.is_file()
            ],
            "cache_candidates": [str(path) for path in caches],
        }
    return {"runtime": runtime_info(), "data_dir": str(data_dir),
            "preprocess_dir": str(preprocess_dir) if preprocess_dir else None,
            "datasets": datasets}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="dataset basenames, e.g. Davis DrugBank")
    parser.add_argument("--data-dir", default=os.environ.get("DTI_DATA_DIR", "DataSets"),
                        help="directory containing <dataset>.txt")
    parser.add_argument("--preprocess-dir", default=os.environ.get("DTI_PREPROCESS_DIR"),
                        help="directory containing subpocket cache files")
    parser.add_argument("--output", default=None, help="optional JSON output path; omitted means stdout only")
    parser.add_argument("--max-list", type=int, default=20, help="maximum example names retained per category")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_list < 0:
        raise SystemExit("--max-list must be non-negative")
    report = build_audit(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
    print(encoded)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
