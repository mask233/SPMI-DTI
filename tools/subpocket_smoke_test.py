#!/usr/bin/env python3
"""CPU smoke test for fixed subpocket slots, masking, and optimizer registration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers import RESCALWithSubpocket  # noqa: E402
from layers_subpocket import SubpocketEncoder  # noqa: E402
from model import MIFDTI  # noqa: E402


FEATURE_DIM = 31


def real_graph(pocket_number: int, nodes: int = 4) -> Data:
    x = torch.randn(nodes, FEATURE_DIM)
    if nodes > 1:
        source = torch.arange(nodes - 1, dtype=torch.long)
        target = source + 1
        edge_index = torch.cat([
            torch.stack([source, target]),
            torch.stack([target, source]),
        ], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        pocket_number=torch.full((nodes,), float(pocket_number)),
    )


def virtual_graph(fill: float = 0.0) -> Data:
    return Data(
        x=torch.full((1, FEATURE_DIM), fill, dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        pocket_number=torch.tensor([-1.0]),
    )


def main() -> int:
    torch.manual_seed(7)
    batch_size, pocket_num, hidden_dim = 2, 3, 8
    sample_slots = [
        [real_graph(1), real_graph(2), virtual_graph()],
        [virtual_graph(), virtual_graph(), virtual_graph()],
    ]

    # Exercise the actual nested PyG collation shape produced by the training
    # loader, then exercise MIFDTI's real preparation method without creating
    # the full model.
    outer = [
        Data(cls_y=torch.tensor(i), subpocket_data=Batch.from_data_list(slots))
        for i, slots in enumerate(sample_slots)
    ]
    outer_batch = next(iter(DataLoader(outer, batch_size=batch_size, shuffle=False)))
    holder = SimpleNamespace(pocket_num=pocket_num)
    prepared = MIFDTI._prepare_subpocket_batch(
        holder, outer_batch.subpocket_data, torch.device("cpu"), batch_size
    )
    graph_count = int(prepared.ptr.numel() - 1)
    assert graph_count == batch_size * pocket_num
    positions = prepared.pocket_number[prepared.ptr[:-1]].view(batch_size, pocket_num)
    assert positions.tolist() == [[1.0, 2.0, -1.0], [-1.0, -1.0, -1.0]]

    encoder = SubpocketEncoder(
        input_dim=FEATURE_DIM,
        hidden_dim=16,
        output_dim=hidden_dim,
        num_layers=2,
        pocket_num=pocket_num,
        pooling=False,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(encoder.input_proj.weight) in optimizer_param_ids

    encoder.train()
    embeddings, pos_index = encoder(prepared, batch_size=batch_size)
    padding_mask = pos_index.eq(-1)
    assert embeddings.shape == (batch_size, pocket_num, hidden_dim)
    assert torch.isfinite(embeddings).all()
    assert torch.count_nonzero(embeddings[padding_mask]).item() == 0
    loss = embeddings[~padding_mask].square().mean()
    optimizer.zero_grad()
    loss.backward()
    gradient = encoder.input_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0

    # Padding values are ignored before graph convolutions and BatchNorm.
    encoder.eval()
    changed_slots = [
        [sample_slots[0][0], sample_slots[0][1], virtual_graph(fill=1e6)],
        [virtual_graph(fill=-1e6), virtual_graph(fill=3e6), virtual_graph(fill=9e6)],
    ]
    changed = Batch.from_data_list([graph for slots in changed_slots for graph in slots])
    with torch.no_grad():
        original_embeddings, _ = encoder(prepared, batch_size=batch_size)
        changed_embeddings, _ = encoder(changed, batch_size=batch_size)
    assert torch.allclose(original_embeddings, changed_embeddings, atol=1e-6, rtol=1e-6)

    fusion = RESCALWithSubpocket(
        n_features=hidden_dim, depth=2, num_heads=2, dropout=0.0
    ).eval()
    drug = torch.randn(batch_size, 2, hidden_dim)
    protein = torch.randn(batch_size, 2, hidden_dim)
    pockets = torch.randn(batch_size, pocket_num, hidden_dim)
    mask = torch.tensor([[False, True, True], [True, True, True]])
    changed_pockets = pockets.clone()
    changed_pockets[mask] = torch.randn_like(changed_pockets[mask]) * 1e6
    with torch.no_grad():
        scores = fusion(drug, protein, pockets, mask)
        changed_scores = fusion(drug, protein, changed_pockets, mask)
        base_scores = fusion._base_scores(drug, protein)
    assert torch.isfinite(scores).all()
    assert torch.allclose(scores, changed_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(scores[1], base_scores[1], atol=1e-6, rtol=1e-6)

    print(json.dumps({
        "status": "ok",
        "torch": torch.__version__,
        "device": "cpu",
        "outer_subpocket_container": type(outer_batch.subpocket_data).__name__,
        "graph_slots": graph_count,
        "input_proj_gradient_l1": float(gradient.abs().sum()),
        "all_padding_finite": True,
        "padding_invariant": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
