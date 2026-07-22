import unittest

import torch
from torch_geometric.data import Data

from models import PDGCNConfig
from training.graph_utils import graph_explicit_source_delta


class CancelExplicitHeatSourceTests(unittest.TestCase):
    def test_disabled_explicit_source_returns_zero_with_nonzero_heat_flux(self):
        graph = Data(
            x=torch.zeros(3, 7),
            q_surface_star=torch.tensor([[0.0], [0.5], [1.0]]),
        )
        graph.num_nodes = 3
        config = PDGCNConfig(
            use_explicit_heat_source=False,
            source_coefficient=2.0,
            dt_star=0.5,
        )

        delta = graph_explicit_source_delta(graph, config)

        self.assertTrue(torch.equal(delta, torch.zeros(3, 1)))

    def test_enabled_explicit_source_behavior_is_unchanged(self):
        graph = Data(
            x=torch.zeros(3, 7),
            q_surface_star=torch.tensor([[0.0], [0.5], [1.0]]),
        )
        graph.num_nodes = 3
        config = PDGCNConfig(
            use_explicit_heat_source=True,
            source_coefficient=2.0,
            dt_star=0.5,
        )

        delta = graph_explicit_source_delta(graph, config)

        self.assertTrue(torch.allclose(delta, graph.q_surface_star))

    def test_switch_requires_boolean_value(self):
        with self.assertRaisesRegex(ValueError, "use_explicit_heat_source must be a boolean"):
            PDGCNConfig(use_explicit_heat_source=0)


if __name__ == "__main__":
    unittest.main()
