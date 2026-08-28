#!/usr/bin/env python3
"""Safely deduplicate DTI rows by the model-visible (SMILES, sequence) pair.

The default mode is read-only and prints a JSON report. Consistent duplicate
groups retain their first row in source order. Any group containing both
labels blocks all writes. Use ``--apply`` only after reviewing the dry run; it
creates a timestamped backup and atomically replaces the input file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def model_pair_sha256(smiles: str, sequence: str) -> str:
    return hashlib.sha256(f"{smiles}\0{sequence}".encode("utf-8")).hexdigest()


def parse_dataset(path: Path) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 5 or fields[-1] not in {"0", "1"}:
            malformed.append({"line": line_number, "text": stripped[:240]})
            continue
        drug_id, protein_id, smiles, sequence, label = (
            fields[-5], fields[-4], fields[-3], fields[-2], fields[-1]
        )
        records.append({
            "line": line_number,
            "text": stripped,
            "drug_id": drug_id,
            "protein_id": protein_id,
            "smiles": smiles,
            "sequence": sequence,
            "label": label,
            "pair_sha256": model_pair_sha256(smiles, sequence),
        })
    return raw, records, malformed


def dataset_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(record["label"] for record in records)
    return {
        "rows": len(records),
        "labels": dict(sorted(labels.items())),
        "unique_drug_ids": len({record["drug_id"] for record in records}),
        "unique_protein_ids": len({record["protein_id"] for record in records}),
        "unique_smiles": len({record["smiles"] for record in records}),
        "unique_sequences": len({record["sequence"] for record in records}),
        "unique_model_pairs": len({record["pair_sha256"] for record in records}),
    }


def build_plan(records: list[dict[str, Any]], max_examples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["pair_sha256"]].append(record)

    kept: list[dict[str, Any]] = []
    duplicate_groups: list[list[dict[str, Any]]] = []
    conflict_groups: list[list[dict[str, Any]]] = []
    for group in groups.values():
        labels = {record["label"] for record in group}
        if len(labels) > 1:
            conflict_groups.append(group)
            continue
        kept.append(group[0])
        if len(group) > 1:
            duplicate_groups.append(group)

    # Restore original row order after dictionary grouping.
    kept.sort(key=lambda record: record["line"])

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        first = group[0]
        consistent = len({record["label"] for record in group}) == 1
        return {
            "pair_sha256": first["pair_sha256"],
            "label_set": sorted({record["label"] for record in group}),
            "source_lines": [record["line"] for record in group],
            "kept_line": first["line"] if consistent else None,
            "removed_lines": [
                record["line"] for record in (group[1:] if consistent else group)
            ],
            "external_pairs": [
                f"{record['drug_id']}::{record['protein_id']}" for record in group
            ],
            "smiles_prefix": first["smiles"][:100],
            "sequence_sha256": hashlib.sha256(
                first["sequence"].encode("utf-8")
            ).hexdigest(),
        }

    plan = {
        "pair_key": "SMILES + protein sequence",
        "duplicate_group_count": len(duplicate_groups),
        "consistent_duplicate_rows_removed": sum(len(group) - 1 for group in duplicate_groups),
        "conflicting_group_count": len(conflict_groups),
        "conflicting_rows": sum(len(group) for group in conflict_groups),
        "duplicate_group_examples": [
            summarize(group) for group in duplicate_groups[:max_examples]
        ],
        "conflicting_group_examples": [
            summarize(group) for group in conflict_groups[:max_examples]
        ],
    }
    return kept, plan


def encoded_dataset(records: list[dict[str, Any]]) -> bytes:
    return ("\n".join(record["text"] for record in records) + "\n").encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="DataSets/DrugBank.txt", help="five-column DTI dataset")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", help="write a new cleaned file; never overwrites unless --force-output")
    destination.add_argument("--apply", action="store_true", help="backup and atomically replace --input")
    parser.add_argument("--force-output", action="store_true", help="allow replacing an existing --output file")
    parser.add_argument("--expect-removed", type=int, help="block writes unless exactly this many rows are removed")
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument("--max-examples", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.force_output and not args.output:
        raise ValueError("--force-output requires --output")
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        protected_paths = {input_path}
        if args.output:
            protected_paths.add(Path(args.output).expanduser().resolve())
        if report_path in protected_paths:
            raise ValueError("--report must not be the input or cleaned output path")

    raw, records, malformed = parse_dataset(input_path)
    kept, plan = build_plan(records, args.max_examples)
    cleaned = encoded_dataset(kept)
    removed = plan["consistent_duplicate_rows_removed"]
    report: dict[str, Any] = {
        "status": "dry_run",
        "input": str(input_path),
        "input_sha256": sha256_bytes(raw),
        "output_sha256": sha256_bytes(cleaned),
        "malformed_row_count": len(malformed),
        "malformed_examples": malformed[:args.max_examples],
        "before": dataset_stats(records),
        "after": dataset_stats(kept),
        **plan,
        "backup": None,
        "written_path": None,
    }

    blocked_reason = None
    if malformed:
        blocked_reason = "malformed rows found"
    elif plan["conflicting_group_count"]:
        blocked_reason = "conflicting labels found for identical model inputs"
    elif args.expect_removed is not None and removed != args.expect_removed:
        blocked_reason = (
            f"expected {args.expect_removed} removed rows, observed {removed}"
        )

    wants_write = bool(args.apply or args.output)
    exit_code = 0
    if blocked_reason:
        report["status"] = "blocked"
        report["blocked_reason"] = blocked_reason
        exit_code = 2
    elif args.apply:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = input_path.with_name(
            f"{input_path.name}.pre_model_pair_dedup_{timestamp}.bak"
        )
        shutil.copy2(input_path, backup)
        atomic_write(input_path, cleaned)
        report["status"] = "applied"
        report["backup"] = str(backup)
        report["written_path"] = str(input_path)
    elif args.output:
        output_path = Path(args.output).expanduser().resolve()
        if output_path == input_path:
            report["status"] = "blocked"
            report["blocked_reason"] = "use --apply to replace the input file"
            exit_code = 2
        elif output_path.exists() and not args.force_output:
            report["status"] = "blocked"
            report["blocked_reason"] = (
                f"output already exists: {output_path}; use --force-output to replace it"
            )
            exit_code = 2
        else:
            atomic_write(output_path, cleaned)
            report["status"] = "written"
            report["written_path"] = str(output_path)
    elif wants_write:
        raise AssertionError("unreachable write mode")

    encoded_report = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(encoded_report, end="")
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        atomic_write(report_path, encoded_report.encode("utf-8"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
