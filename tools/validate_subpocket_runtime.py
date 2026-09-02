#!/usr/bin/env python3
"""Exercise every cached subpocket Batch through the stable split path."""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.subpocket_batch import (  # noqa: E402
    combine_subpocket_graphs,
    prepare_flat_subpocket_batch,
    split_subpocket_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--expected-slots", type=int, default=30)
    parser.add_argument("--outer-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    if args.repetitions < 1 or args.expected_slots < 1 or args.outer_batch_size < 1:
        raise ValueError("Repetitions, slots and outer batch size must be positive.")
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    try:
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(args.cache, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError(f"Expected a dict cache, got {type(cache)!r}.")

    checked = 0
    checked_outer_batches = 0
    none_entries = 0
    for repetition in range(1, args.repetitions + 1):
        print(f"[CACHE CHECK] repetition={repetition}/{args.repetitions}", flush=True)
        outer_graphs = []
        for key, value in cache.items():
            if value is None:
                none_entries += int(repetition == 1)
                continue
            if not isinstance(value, Batch):
                raise TypeError(f"Cache entry {key!r} is {type(value)!r}, expected Batch.")
            graphs = split_subpocket_batch(value)
            if len(graphs) != args.expected_slots:
                raise RuntimeError(
                    f"Cache entry {key!r} has {len(graphs)} graph slots; "
                    f"expected {args.expected_slots}."
                )
            # Exercise the exact model-side replacement for
            # Batch.from_data_list as well as the cache-side split.
            rebuilt = combine_subpocket_graphs(graphs)
            if rebuilt.ptr.numel() != args.expected_slots + 1:
                raise RuntimeError(f"Cache entry {key!r} rebuilt with an invalid ptr.")
            outer_graphs.extend(graphs)
            if len(outer_graphs) == args.outer_batch_size * args.expected_slots:
                outer_batch = combine_subpocket_graphs(outer_graphs)
                if outer_batch.ptr.numel() != len(outer_graphs) + 1:
                    raise RuntimeError("Full outer batch rebuilt with an invalid ptr.")
                optimized_batch = prepare_flat_subpocket_batch(
                    outer_batch,
                    expected_graphs=len(outer_graphs),
                    device="cpu",
                    validate_values=True,
                )
                for field in ("x", "edge_index", "pocket_number", "batch", "ptr"):
                    if not torch.equal(getattr(outer_batch, field), getattr(optimized_batch, field)):
                        raise RuntimeError(f"Optimized outer batch changed {field}.")
                checked_outer_batches += 1
                outer_graphs = []
            checked += 1
        if outer_graphs:
            outer_batch = combine_subpocket_graphs(outer_graphs)
            if outer_batch.ptr.numel() != len(outer_graphs) + 1:
                raise RuntimeError("Final partial outer batch rebuilt with an invalid ptr.")
            optimized_batch = prepare_flat_subpocket_batch(
                outer_batch,
                expected_graphs=len(outer_graphs),
                device="cpu",
                validate_values=True,
            )
            for field in ("x", "edge_index", "pocket_number", "batch", "ptr"):
                if not torch.equal(getattr(outer_batch, field), getattr(optimized_batch, field)):
                    raise RuntimeError(f"Optimized final batch changed {field}.")
            checked_outer_batches += 1
        print(f"[CACHE CHECK] repetition={repetition} completed", flush=True)

    print(json.dumps({
        "status": "ok",
        "cache": str(args.cache),
        "entries": len(cache),
        "none_entries": none_entries,
        "repetitions": args.repetitions,
        "checked_entry_passes": checked,
        "outer_batch_size": args.outer_batch_size,
        "checked_outer_batches": checked_outer_batches,
        "runtime_method": "prevalidated_flat_tensor_batch_without_per_sample_split",
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
