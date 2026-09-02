#!/usr/bin/env python3
"""Compare legacy and optimized runtime paths on one identical real batch."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# Required by deterministic cuBLAS algorithms on CUDA >= 10.2.  Set it before
# importing torch or any dependency that may initialize a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import hyperparameter  # noqa: E402
from model import (  # noqa: E402
    MIFDTI,
    get_m2p_edge_from_batch,
    get_m2p_edge_from_batch_dense,
)
from utils.dataset import ProteinMoleculeDataset  # noqa: E402
from utils.subpocket_batch import (  # noqa: E402
    combine_subpocket_graphs,
    prepare_subpocket_runtime_batch,
    split_subpocket_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["Davis", "DrugBank", "BIOSNAP", "BD2D"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--preprocess-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enforce deterministic PyTorch/CUDA algorithms for strict equivalence "
            "checking (default: enabled)"
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_verification_determinism(enabled: bool) -> None:
    """Use the strictest practical execution mode for equivalence checks."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(enabled)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not enabled
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not enabled
    torch.use_deterministic_algorithms(bool(enabled), warn_only=False)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_state(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    weight_parameters = []
    bias_parameters = []
    for name, parameter in model.named_parameters():
        (bias_parameters if "bias" in name else weight_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": weight_parameters, "weight_decay": weight_decay},
            {"params": bias_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training_steps(
        *,
        initial_state,
        batch,
        hp,
        device: torch.device,
        seed: int,
        steps: int,
        optimized: bool,
        class_weight,
):
    model = MIFDTI(
        device=device,
        use_subpocket=hp.use_subpocket,
        pocket_num=hp.pocket_num,
        debug_data_validation=False,
        runtime_optimization=optimized,
    )
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = make_optimizer(model, hp.Learning_rate, hp.weight_decay)
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_seconds = 0.0
    logits_per_step = []
    losses = []
    gradients = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        started = time.perf_counter()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.cls_y, weight=class_weight)
        sync(device)
        forward_done = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        sync(device)
        backward_done = time.perf_counter()
        gradients = {
            name: (None if parameter.grad is None else parameter.grad.detach().cpu().clone())
            for name, parameter in model.named_parameters()
        }
        optimizer.step()
        sync(device)
        finished = time.perf_counter()
        logits_per_step.append(logits.detach().cpu().clone())
        losses.append(loss.detach().cpu().clone())
        forward_seconds += forward_done - started
        backward_seconds += backward_done - forward_done
        optimizer_seconds += finished - backward_done
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    result = {
        "logits": torch.stack(logits_per_step),
        "loss": torch.stack(losses),
        "gradients": gradients,
        "updated_state": clone_state(model.state_dict()),
        "timing": {
            "steps": steps,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
            "total_seconds": forward_seconds + backward_seconds + optimizer_seconds,
            "peak_cuda_memory_bytes": peak_memory,
        },
    }
    del optimizer, model, logits, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def tensor_comparison(reference, candidate):
    if reference is None or candidate is None:
        return {
            "equal": reference is None and candidate is None,
            "max_abs_diff": None,
        }
    equal = torch.equal(reference, candidate)
    if reference.shape != candidate.shape:
        return {"equal": False, "max_abs_diff": None}
    if reference.numel() == 0:
        max_abs_diff = 0.0
    elif reference.is_floating_point() or reference.is_complex():
        max_abs_diff = float((reference - candidate).abs().max())
    else:
        max_abs_diff = 0.0 if equal else None
    return {"equal": bool(equal), "max_abs_diff": max_abs_diff}


def mapping_comparison(reference, candidate):
    keys_equal = set(reference) == set(candidate)
    keys = sorted(set(reference) | set(candidate))
    unequal = {}
    for key in keys:
        if key not in reference or key not in candidate:
            unequal[key] = {"equal": False, "max_abs_diff": None}
            continue
        comparison = tensor_comparison(reference[key], candidate[key])
        if not comparison["equal"]:
            unequal[key] = comparison
    return {
        "equal": bool(keys_equal and not unequal),
        "unequal_count": len(unequal),
        "max_abs_diff": max(
            (
                value["max_abs_diff"]
                for value in unequal.values()
                if value["max_abs_diff"] is not None
            ),
            default=0.0,
        ),
        "first_unequal": dict(list(unequal.items())[:10]),
    }


def compare_training_runs(reference, candidate):
    return {
        "train_logits": tensor_comparison(reference["logits"], candidate["logits"]),
        "loss": tensor_comparison(reference["loss"], candidate["loss"]),
        "all_parameter_gradients": mapping_comparison(
            reference["gradients"], candidate["gradients"]
        ),
        "state_after_optimizer_step": mapping_comparison(
            reference["updated_state"], candidate["updated_state"]
        ),
    }


def all_training_checks_equal(checks) -> bool:
    return all(check["equal"] for check in checks.values())


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for training-mode BatchNorm.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    configure_verification_determinism(args.deterministic)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    hp = hyperparameter(
        args.dataset, data_dir=args.data_dir, preprocess_dir=args.preprocess_dir
    )
    data_dir = Path(hp.data_dir)
    preprocess_dir = Path(hp.preprocess_dir)
    dataset_path = require_file(data_dir / f"{args.dataset}.txt")
    protein_path = require_file(preprocess_dir / f"{args.dataset}-protein.pkl")
    ligand_path = require_file(preprocess_dir / f"{args.dataset}-ligand-hi.pkl")
    subpocket_path = require_file(preprocess_dir / f"{args.dataset}-subpocket.pt")

    rows = [
        line.strip()
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.batch_size:
        raise ValueError("Dataset has fewer rows than --batch-size.")
    dataset = ProteinMoleculeDataset(
        rows[:args.batch_size],
        joblib.load(ligand_path),
        joblib.load(protein_path),
        device=device,
        use_3d=False,
        pdb_dir=hp.pdb_dir,
        prot_3d_obj=None,
        use_subpocket=True,
        subpocket_dir=hp.subpocket_dir,
        pocket_num=hp.pocket_num,
        subpocket_cache_path=str(subpocket_path),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["mol_x", "clique_x", "prot_node_aa"],
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader)).to(device)

    def legacy_subpocket_graphs(raw_data):
        if isinstance(raw_data, Batch):
            return split_subpocket_batch(raw_data)
        if isinstance(raw_data, (list, tuple)):
            graphs = []
            for item in raw_data:
                graphs.extend(legacy_subpocket_graphs(item))
            return graphs
        raise TypeError(f"Unsupported subpocket container: {type(raw_data)!r}")

    legacy_subpocket = combine_subpocket_graphs(
        legacy_subpocket_graphs(batch.subpocket_data), device=device
    )
    optimized_subpocket = prepare_subpocket_runtime_batch(
        batch.subpocket_data,
        batch_size=int(batch.cls_y.size(0)),
        graphs_per_sample=hp.pocket_num,
        device=device,
        validate_values=True,
    )
    subpocket_check = mapping_comparison(
        {
            field: getattr(legacy_subpocket, field).detach().cpu()
            for field in ("x", "edge_index", "pocket_number", "batch", "ptr")
        },
        {
            field: getattr(optimized_subpocket, field).detach().cpu()
            for field in ("x", "edge_index", "pocket_number", "batch", "ptr")
        },
    )
    del legacy_subpocket, optimized_subpocket

    dense_edges = get_m2p_edge_from_batch_dense(
        batch.mol_x_batch, batch.prot_node_aa_batch,
        node_level=getattr(batch, "mol_node_levels", None),
    )
    optimized_edges = get_m2p_edge_from_batch(
        batch.mol_x_batch, batch.prot_node_aa_batch,
        node_level=getattr(batch, "mol_node_levels", None),
    )
    edge_check = tensor_comparison(dense_edges.detach().cpu(), optimized_edges.detach().cpu())

    set_seed(args.seed)
    base_model = MIFDTI(
        device="cpu", use_subpocket=True, pocket_num=hp.pocket_num,
        runtime_optimization=False,
    )
    if args.checkpoint is not None:
        base_model.load_state_dict(load_state(require_file(args.checkpoint)), strict=True)
    else:
        for parameter in base_model.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
    initial_state = clone_state(base_model.state_dict())
    del base_model

    if args.dataset == "Davis":
        class_weight = torch.tensor([0.3, 0.7], dtype=torch.float32, device=device)
    else:
        class_weight = None

    reference = run_training_steps(
        initial_state=initial_state,
        batch=batch,
        hp=hp,
        device=device,
        seed=args.seed,
        steps=args.steps,
        optimized=False,
        class_weight=class_weight,
    )
    candidate = run_training_steps(
        initial_state=initial_state,
        batch=batch,
        hp=hp,
        device=device,
        seed=args.seed,
        steps=args.steps,
        optimized=True,
        class_weight=class_weight,
    )
    # A same-path replay distinguishes a genuine implementation difference
    # from CUDA/PyG operations that are not bitwise repeatable even after the
    # documented deterministic settings have been requested.
    reference_replay = run_training_steps(
        initial_state=initial_state,
        batch=batch,
        hp=hp,
        device=device,
        seed=args.seed,
        steps=args.steps,
        optimized=False,
        class_weight=class_weight,
    )

    cross_path_checks = compare_training_runs(reference, candidate)
    repeatability_checks = compare_training_runs(reference, reference_replay)
    structural_ok = subpocket_check["equal"] and edge_check["equal"]
    cross_path_ok = all_training_checks_equal(cross_path_checks)
    repeatability_ok = all_training_checks_equal(repeatability_checks)
    ok = structural_ok and cross_path_ok and repeatability_ok
    if ok:
        diagnosis = "strict_equivalence_confirmed"
    elif not repeatability_ok:
        diagnosis = "same_path_not_bitwise_repeatable"
    else:
        diagnosis = "optimized_path_differs_from_repeatable_reference"
    report = {
        "status": "ok" if ok else ("inconclusive" if not repeatability_ok else "error"),
        "diagnosis": diagnosis,
        "dataset": args.dataset,
        "device": str(device),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "verification_mode": {
            "deterministic_algorithms": bool(args.deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_enabled": not bool(args.deterministic),
        },
        "strict_checks": {
            "subpocket_tensors_and_order": subpocket_check,
            "m2p_edges_and_order": edge_check,
            **cross_path_checks,
        },
        "same_legacy_path_replay_checks": repeatability_checks,
        "timing": {
            "legacy": reference["timing"],
            "optimized": candidate["timing"],
            "legacy_replay": reference_replay["timing"],
            "note": "diagnostic only; execution order includes warm-up effects",
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
