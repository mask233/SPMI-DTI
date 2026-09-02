from __future__ import annotations

import unittest

try:
    import torch
    from torch_geometric.data import Batch, Data
    from utils.subpocket_batch import (
        combine_subpocket_graphs,
        prepare_flat_subpocket_batch,
        prepare_subpocket_runtime_batch,
        split_subpocket_batch,
    )
except (ImportError, OSError):
    torch = Batch = Data = combine_subpocket_graphs = prepare_flat_subpocket_batch = None
    prepare_subpocket_runtime_batch = None
    split_subpocket_batch = None


@unittest.skipIf(torch is None, "working PyTorch/PyG runtime is unavailable")
class StableSubpocketBatchTests(unittest.TestCase):
    def test_manual_split_preserves_graphs_and_is_repeatable(self):
        graphs = [
            Data(
                x=torch.arange(nodes * 31, dtype=torch.float32).view(nodes, 31),
                edge_index=(
                    torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                    if nodes > 1 else torch.zeros((2, 0), dtype=torch.long)
                ),
                pocket_number=torch.full((nodes,), float(index + 1)),
            )
            for index, nodes in enumerate([2, 1, 3])
        ]
        cached = Batch.from_data_list(graphs)
        for _ in range(5):
            recovered = split_subpocket_batch(cached)
            self.assertEqual(len(recovered), len(graphs))
            for expected, actual in zip(graphs, recovered):
                self.assertTrue(torch.equal(expected.x, actual.x))
                self.assertTrue(torch.equal(expected.edge_index, actual.edge_index))
                self.assertTrue(torch.equal(expected.pocket_number, actual.pocket_number))
            rebuilt = combine_subpocket_graphs(recovered)
            self.assertTrue(torch.equal(cached.x, rebuilt.x))
            self.assertTrue(torch.equal(cached.edge_index, rebuilt.edge_index))
            self.assertTrue(torch.equal(cached.pocket_number, rebuilt.pocket_number))
            self.assertTrue(torch.equal(cached.ptr, rebuilt.ptr))

    def test_cross_graph_edge_is_rejected(self):
        cached = Batch.from_data_list([
            Data(x=torch.ones((1, 31)), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 pocket_number=torch.ones(1)),
            Data(x=torch.ones((1, 31)), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 pocket_number=torch.ones(1)),
        ])
        cached.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "cross-graph"):
            split_subpocket_batch(cached)

    def test_flat_prepare_matches_split_and_recombine(self):
        graphs = [
            Data(
                x=torch.arange(nodes * 31, dtype=torch.float32).view(nodes, 31),
                edge_index=(
                    torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                    if nodes > 1 else torch.zeros((2, 0), dtype=torch.long)
                ),
                pocket_number=torch.full((nodes,), float(index + 1)),
            )
            for index, nodes in enumerate([2, 1, 3])
        ]
        raw = combine_subpocket_graphs(graphs)
        legacy = combine_subpocket_graphs(split_subpocket_batch(raw))
        optimized = prepare_flat_subpocket_batch(
            raw, expected_graphs=len(graphs), device="cpu", validate_values=True
        )
        for field in ("x", "edge_index", "pocket_number", "batch", "ptr"):
            self.assertTrue(torch.equal(getattr(legacy, field), getattr(optimized, field)))

    def test_per_sample_batch_list_matches_legacy_order(self):
        graphs = [
            Data(
                x=(torch.arange(nodes * 31, dtype=torch.float32).view(nodes, 31)
                   + graph_index * 1000),
                edge_index=(
                    torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                    if nodes > 1 else torch.zeros((2, 0), dtype=torch.long)
                ),
                pocket_number=torch.full((nodes,), float(graph_index + 1)),
            )
            for graph_index, nodes in enumerate([2, 1, 3, 1, 2, 4])
        ]
        per_sample = [
            Batch.from_data_list(graphs[:3]),
            Batch.from_data_list(graphs[3:]),
        ]
        legacy = combine_subpocket_graphs(
            split_subpocket_batch(per_sample[0])
            + split_subpocket_batch(per_sample[1])
        )
        optimized = prepare_subpocket_runtime_batch(
            per_sample,
            batch_size=2,
            graphs_per_sample=3,
            device="cpu",
            validate_values=True,
        )
        for field in ("x", "edge_index", "pocket_number", "batch", "ptr"):
            self.assertTrue(torch.equal(getattr(legacy, field), getattr(optimized, field)))

    def test_per_sample_batch_list_rejects_wrong_shape(self):
        item = Batch.from_data_list([
            Data(
                x=torch.ones((1, 31)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                pocket_number=torch.ones(1),
            )
        ])
        with self.assertRaisesRegex(RuntimeError, "samples; expected"):
            prepare_subpocket_runtime_batch(
                [item], batch_size=2, graphs_per_sample=1, device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
