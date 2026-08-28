"""Stable helpers for the narrow subpocket PyG cache schema."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data


def combine_subpocket_graphs(graphs: list[Data], device=None) -> Batch:
    """Build the narrow runtime Batch directly, without PyG collation."""
    if not graphs:
        raise ValueError("At least one subpocket graph is required.")

    feature_width = None
    x_parts = []
    edge_parts = []
    pocket_parts = []
    batch_parts = []
    boundaries = [0]
    node_offset = 0
    for graph_index, graph in enumerate(graphs):
        if not isinstance(graph, Data):
            raise TypeError(f"Subpocket graph {graph_index} is {type(graph)!r}, expected Data.")
        x = getattr(graph, "x", None)
        edge_index = getattr(graph, "edge_index", None)
        pocket_number = getattr(graph, "pocket_number", None)
        if x is None or x.dim() != 2 or x.size(0) < 1:
            raise ValueError(f"Subpocket graph {graph_index} has an invalid x tensor.")
        if feature_width is None:
            feature_width = x.size(1)
        elif x.size(1) != feature_width:
            raise ValueError("Subpocket graphs do not share one feature width.")
        if edge_index is None or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Subpocket graph {graph_index} has an invalid edge_index.")
        if edge_index.numel() and (
                edge_index.min().item() < 0 or edge_index.max().item() >= x.size(0)):
            raise ValueError(f"Subpocket graph {graph_index} has an out-of-range edge.")
        if pocket_number is None or pocket_number.numel() != x.size(0):
            raise ValueError(f"Subpocket graph {graph_index} has invalid pocket_number values.")
        if device is not None:
            x = x.to(device)
            edge_index = edge_index.to(device)
            pocket_number = pocket_number.to(device)

        x_parts.append(x)
        if edge_index.numel():
            edge_parts.append(edge_index + node_offset)
        pocket_parts.append(pocket_number.reshape(-1))
        batch_parts.append(torch.full(
            (x.size(0),), graph_index, dtype=torch.long, device=x.device
        ))
        node_offset += int(x.size(0))
        boundaries.append(node_offset)

    first_x = x_parts[0]
    merged = Batch(
        x=torch.cat(x_parts, dim=0),
        edge_index=(
            torch.cat(edge_parts, dim=1)
            if edge_parts else torch.zeros((2, 0), dtype=torch.long, device=first_x.device)
        ),
        pocket_number=torch.cat(pocket_parts, dim=0),
        batch=torch.cat(batch_parts, dim=0),
        ptr=torch.tensor(boundaries, dtype=torch.long, device=first_x.device),
    )
    # Avoid any need for PyG to infer this through its dynamic storage views.
    merged._num_graphs = len(graphs)
    return merged


def split_subpocket_batch(batch_data: Batch) -> list[Data]:
    """Split a cached Batch without calling PyG ``to_data_list``.

    Some serialized PyG Batch objects crash inside ``separate``/``view`` when
    repeatedly converted with ``to_data_list``. Subpocket caches have a much
    narrower contract (x, edge_index, pocket_number and ptr), so reconstructing
    the graphs directly from node boundaries is both deterministic and safer.
    """
    if not isinstance(batch_data, Batch):
        raise TypeError(f"Expected a PyG Batch, got {type(batch_data)!r}.")
    x = getattr(batch_data, "x", None)
    edge_index = getattr(batch_data, "edge_index", None)
    pocket_number = getattr(batch_data, "pocket_number", None)
    ptr = getattr(batch_data, "ptr", None)
    if x is None or x.dim() != 2:
        raise ValueError("Cached subpocket Batch has no two-dimensional x tensor.")
    if edge_index is None or edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("Cached subpocket Batch has an invalid edge_index tensor.")
    if pocket_number is None or pocket_number.numel() != x.size(0):
        raise ValueError("Cached subpocket Batch has an invalid pocket_number tensor.")
    if ptr is None or ptr.dim() != 1 or ptr.numel() < 2:
        raise ValueError("Cached subpocket Batch has no valid ptr graph boundaries.")

    boundaries = [int(value) for value in ptr.detach().cpu().tolist()]
    if boundaries[0] != 0 or boundaries[-1] != x.size(0):
        raise ValueError(
            f"Cached subpocket ptr must span 0..{x.size(0)}, got "
            f"{boundaries[0]}..{boundaries[-1]}."
        )
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Every cached subpocket graph must contain at least one node.")

    if edge_index.numel():
        if edge_index.min().item() < 0 or edge_index.max().item() >= x.size(0):
            raise ValueError("Cached subpocket edge_index contains an out-of-range node.")
        # Assign every edge to its source graph once and reject cross-graph
        # edges.  Bucketization avoids scanning all edges once per graph.
        upper_bounds = ptr[1:].to(device=edge_index.device)
        source_graph = torch.bucketize(edge_index[0], upper_bounds, right=True)
        target_graph = torch.bucketize(edge_index[1], upper_bounds, right=True)
        if not torch.equal(source_graph, target_graph):
            raise ValueError("Cached subpocket Batch contains a cross-graph edge.")
    else:
        source_graph = torch.empty(0, dtype=torch.long, device=edge_index.device)

    graphs: list[Data] = []
    for graph_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        edge_mask = source_graph.eq(graph_index)
        graph_edges = edge_index[:, edge_mask] - start
        graphs.append(Data(
            x=x[start:end].clone(),
            edge_index=graph_edges.clone(),
            pocket_number=pocket_number.reshape(-1)[start:end].clone(),
        ))
    return graphs
