from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
    from model import get_m2p_edge_from_batch, get_m2p_edge_from_batch_dense
except (ImportError, OSError):
    torch = get_m2p_edge_from_batch = get_m2p_edge_from_batch_dense = None


@unittest.skipIf(torch is None, "working PyTorch/PyG runtime is unavailable")
class InteractionEdgeOptimizationTests(unittest.TestCase):
    def test_sparse_builder_preserves_dense_edge_order(self):
        cases = [
            (
                torch.tensor([0, 0, 1, 1, 1]),
                torch.tensor([0, 0, 0, 1, 1, 1]),
                None,
            ),
            (
                torch.tensor([0, 0, 1, 1, 1]),
                torch.tensor([0, 0, 0, 1, 1, 1]),
                torch.tensor([1, 0, 1, 1, 0]),
            ),
            (
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                torch.tensor([0, 0]),
            ),
        ]
        for atom_batch, aa_batch, node_level in cases:
            expected = get_m2p_edge_from_batch_dense(
                atom_batch, aa_batch, node_level=node_level
            )
            actual = get_m2p_edge_from_batch(
                atom_batch, aa_batch, node_level=node_level
            )
            self.assertTrue(torch.equal(expected, actual))

    def test_sparse_builder_matches_random_unsorted_batches(self):
        generator = torch.Generator().manual_seed(20260828)
        for _ in range(20):
            atom_batch = torch.randint(0, 4, (17,), generator=generator)
            aa_batch = torch.randint(0, 4, (23,), generator=generator)
            node_level = torch.randint(0, 3, (17,), generator=generator)
            expected = get_m2p_edge_from_batch_dense(
                atom_batch, aa_batch, node_level=node_level
            )
            actual = get_m2p_edge_from_batch(
                atom_batch, aa_batch, node_level=node_level
            )
            self.assertTrue(torch.equal(expected, actual))


class DatasetDeadFieldTests(unittest.TestCase):
    def test_dataset_no_longer_constructs_unused_m2p_edges(self):
        source = (Path(__file__).resolve().parents[1] / "utils" / "dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("get_m2p_edge(", source)
        self.assertNotIn("'m2p_edge_index'", source)
        self.assertNotIn('"m2p_edge_index"', source)


if __name__ == "__main__":
    unittest.main()
