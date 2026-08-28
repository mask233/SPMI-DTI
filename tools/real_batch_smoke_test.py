#!/usr/bin/env python3
"""Run one real cached batch through forward/backward without writing outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import hyperparameter  # noqa: E402
from model import MIFDTI  # noqa: E402
from utils.dataset import ProteinMoleculeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["Davis", "DrugBank", "BIOSNAP", "BD2D"])
    parser.add_argument("--data-dir", default=None,
                        help="default: DTI_DATA_DIR or ./DataSets")
    parser.add_argument("--preprocess-dir", default=None,
                        help="default: DTI_PREPROCESS_DIR or <data-dir>/Preprocessed")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None, help="default: cuda:0 when available, otherwise cpu")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required smoke-test input not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for training-mode BatchNorm smoke testing.")
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    hp = hyperparameter(
        args.dataset, data_dir=args.data_dir, preprocess_dir=args.preprocess_dir
    )
    data_dir = Path(hp.data_dir)
    preprocess_dir = Path(hp.preprocess_dir)
    dataset_file = require_file(data_dir / f"{args.dataset}.txt")
    protein_file = require_file(preprocess_dir / f"{args.dataset}-protein.pkl")
    ligand_file = require_file(preprocess_dir / f"{args.dataset}-ligand-hi.pkl")
    subpocket_file = require_file(preprocess_dir / f"{args.dataset}-subpocket.pt")

    rows = [line.strip() for line in dataset_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < args.batch_size:
        raise ValueError(f"Dataset has {len(rows)} rows, fewer than batch size {args.batch_size}.")
    protein_dict = joblib.load(protein_file)
    ligand_dict = joblib.load(ligand_file)
    dataset = ProteinMoleculeDataset(
        rows,
        ligand_dict,
        protein_dict,
        device=device,
        use_3d=False,
        pdb_dir=hp.pdb_dir,
        prot_3d_obj=None,
        use_subpocket=True,
        subpocket_dir=hp.subpocket_dir,
        pocket_num=hp.pocket_num,
        subpocket_cache_path=str(subpocket_file),
    )
    real_pocket_rows = []
    for row in rows:
        fields = row.split()
        target, sequence = fields[1], fields[-2]
        cached = dataset.prots_subpocket.get(
            target, dataset.prots_subpocket.get(sequence)
        )
        pocket_number = getattr(cached, "pocket_number", None)
        if pocket_number is not None and pocket_number.ne(-1).any().item():
            real_pocket_rows.append(row)
    if not real_pocket_rows:
        raise AssertionError("No dataset row resolves to a cache entry with a real subpocket.")
    selected_rows = [real_pocket_rows[0]]
    selected_rows.extend(row for row in rows if row != selected_rows[0])
    dataset.pairs = selected_rows[:args.batch_size]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
        drop_last=False,
    )
    batch = next(iter(loader)).to(device)
    model = MIFDTI(
        device=device, use_subpocket=True, pocket_num=hp.pocket_num
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if id(model.subpocket_encoder.input_proj.weight) not in optimizer_param_ids:
        raise AssertionError("subpocket input_proj is missing from optimizer parameters")

    prepared = model._prepare_subpocket_batch(
        batch.subpocket_data, device=device, batch_size=args.batch_size
    )
    positions = prepared.pocket_number[prepared.ptr[:-1]].view(
        args.batch_size, hp.pocket_num
    )
    real_slots = int(positions.ne(-1).sum().item())
    expected_slots = args.batch_size * hp.pocket_num
    if prepared.ptr.numel() - 1 != expected_slots:
        raise AssertionError("real batch did not preserve B x M subpocket slots")

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.cross_entropy(logits, batch.cls_y)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise FloatingPointError("non-finite logits or loss in real-batch smoke test")
    loss.backward()
    gradient = model.subpocket_encoder.input_proj.weight.grad
    if real_slots > 0 and (
            gradient is None
            or not torch.isfinite(gradient).all()
            or gradient.abs().sum().item() == 0):
        raise AssertionError("subpocket input_proj gradient is missing, zero, or non-finite")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = None

    print(json.dumps({
        "status": "ok",
        "dataset": args.dataset,
        "device": str(device),
        "batch_size": args.batch_size,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "subpocket_container": type(batch.subpocket_data).__name__,
        "graph_slots": int(prepared.ptr.numel() - 1),
        "real_slots": real_slots,
        "virtual_slots": expected_slots - real_slots,
        "input_proj_gradient_l1": (
            float(gradient.abs().sum().detach().cpu()) if gradient is not None else None
        ),
        "peak_cuda_memory_bytes": peak_memory,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
