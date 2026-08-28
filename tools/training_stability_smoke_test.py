#!/usr/bin/env python3
"""Stress the real cached training path across repeated full-size batches."""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="0 means a complete epoch")
    parser.add_argument("--validation-batches", type=int, default=5,
                        help="evaluation batches between training epochs; 0 disables validation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None,
                        help="default: cuda:0 when available, otherwise cpu")
    parser.add_argument("--cuda-stability", action="store_true",
                        help="synchronous conservative CUDA path for localization")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required stress-test input not found: {path}")
    return path


def configure_cuda_stability() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "_inductor"):
        torch._inductor.config.triton.cudagraphs = False


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for training-mode BatchNorm.")
    if (args.epochs < 1 or args.max_batches_per_epoch < 0
            or args.validation_batches < 0):
        raise ValueError("Epoch and batch limits must be non-negative, with --epochs positive.")
    if args.cuda_stability:
        configure_cuda_stability()
        print("[STRESS] conservative synchronous CUDA mode enabled", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    hp = hyperparameter(
        args.dataset, data_dir=args.data_dir, preprocess_dir=args.preprocess_dir
    )
    data_dir = Path(hp.data_dir)
    preprocess_dir = Path(hp.preprocess_dir)
    dataset_file = require_file(data_dir / f"{args.dataset}.txt")
    protein_file = require_file(preprocess_dir / f"{args.dataset}-protein.pkl")
    ligand_file = require_file(preprocess_dir / f"{args.dataset}-ligand-hi.pkl")
    subpocket_file = require_file(preprocess_dir / f"{args.dataset}-subpocket.pt")

    rows = [
        line.strip()
        for line in dataset_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.batch_size:
        raise ValueError(f"Dataset has {len(rows)} rows, fewer than batch size {args.batch_size}.")

    print(
        "[STRESS] runtime "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"pyg={package_version('torch-geometric')} "
        f"torch-scatter={package_version('torch-scatter')} "
        f"device={device}",
        flush=True,
    )
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
        drop_last=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    model = MIFDTI(
        device=device, use_subpocket=True, pocket_num=hp.pocket_num
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.Learning_rate)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    completed_steps = 0
    validation_steps = 0
    last_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_steps = 0
        print(f"[STRESS] epoch={epoch}/{args.epochs} DataLoader start", flush=True)
        for batch_index, batch in enumerate(loader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            print(f"[STRESS] epoch={epoch} batch={batch_index} collated", flush=True)
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.cls_y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite output at epoch={epoch}, batch={batch_index}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            last_loss = float(loss.detach().cpu())
            completed_steps += 1
            epoch_steps += 1
            print(
                f"[STRESS] epoch={epoch} batch={batch_index} completed "
                f"loss={last_loss:.6f}",
                flush=True,
            )
            del batch, logits, loss
        if epoch_steps == 0:
            raise RuntimeError(f"No full batch was completed in epoch {epoch}.")
        if args.validation_batches:
            model.eval()
            print(f"[STRESS] epoch={epoch} validation start", flush=True)
            with torch.no_grad():
                for validation_index, batch in enumerate(validation_loader):
                    if validation_index >= args.validation_batches:
                        break
                    batch = batch.to(device)
                    logits = model(batch)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            f"non-finite validation output at epoch={epoch}, "
                            f"batch={validation_index}"
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    validation_steps += 1
                    print(
                        f"[STRESS] epoch={epoch} validation_batch={validation_index} completed",
                        flush=True,
                    )
                    del batch, logits
            model.train()

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    print(json.dumps({
        "status": "ok",
        "dataset": args.dataset,
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_batches_per_epoch": args.max_batches_per_epoch,
        "completed_steps": completed_steps,
        "validation_steps": validation_steps,
        "last_loss": last_loss,
        "cuda_stability": args.cuda_stability,
        "peak_cuda_memory_bytes": peak_memory,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
