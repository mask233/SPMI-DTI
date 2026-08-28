"""Shared helpers for deterministic DTI split manifests.

The training code and the standalone split generator both import this module.
Keeping the row parser, identifiers, and CSV writer in one place prevents a
baseline manifest from silently drifting away from the main-model protocol.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MANIFEST_FIELDS = [
    "split",
    "sample_index",
    "sample_id",
    "external_pair_id",
    "drug_id",
    "protein_id",
    "label",
    "row_sha256",
    "seed",
    "fold",
]


def parse_dti_row(row: str) -> tuple[str, str, str, str, str]:
    """Return the final five whitespace-separated DTI fields."""
    parts = row.split()
    if len(parts) < 5:
        raise ValueError(f"Malformed DTI row: {row[:160]}")
    drug_id, protein_id, smiles, sequence, label = parts[-5:]
    if label not in {"0", "1"}:
        raise ValueError(f"Non-binary DTI label {label!r}: {row[:160]}")
    return drug_id, protein_id, smiles, sequence, label


def model_pair_sha256(smiles: str, sequence: str) -> str:
    """Hash exactly the pair visible to the model."""
    return hashlib.sha256(f"{smiles}\0{sequence}".encode("utf-8")).hexdigest()


def row_sha256(row: str) -> str:
    return hashlib.sha256(row.encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dti_rows(path: str | os.PathLike[str]) -> list[str]:
    """Load non-empty rows and validate their five required fields."""
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        raise ValueError(f"Dataset is empty: {path}")
    # Match RunModel.py exactly: strip only the complete file, not each row.
    rows = content.split("\n")
    for row in rows:
        parse_dti_row(row)
    return rows


def manifest_records(
    split_rows: Mapping[str, Sequence[str]], seed: int, fold: int | None = None
) -> Iterable[dict[str, object]]:
    for split, rows in split_rows.items():
        for sample_index, row in enumerate(rows):
            drug_id, protein_id, smiles, sequence, label = parse_dti_row(row)
            yield {
                "split": split,
                "sample_index": sample_index,
                "sample_id": model_pair_sha256(smiles, sequence),
                "external_pair_id": f"{drug_id}::{protein_id}",
                "drug_id": drug_id,
                "protein_id": protein_id,
                "label": label,
                "row_sha256": row_sha256(row),
                "seed": seed,
                "fold": "" if fold is None else fold,
            }


def save_split_manifest(
    filepath: str | os.PathLike[str],
    split_rows: Mapping[str, Sequence[str]],
    seed: int,
    fold: int | None = None,
) -> None:
    """Write a deterministic ID/hash-only manifest, replacing that one file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_records(split_rows, seed=seed, fold=fold))
