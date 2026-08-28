from __future__ import annotations

import unittest

try:
    import torch
    from torch_geometric.data import Batch, Data
    from utils.subpocket_batch import combine_subpocket_graphs, split_subpocket_batch
except (ImportError, OSError):
    torch = Batch = Data = combine_subpocket_graphs = split_subpocket_batch = None


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


if __name__ == "__main__":
    unittest.main()
