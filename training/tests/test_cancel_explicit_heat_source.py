import unittest

import torch
from torch import nn
from torch_geometric.data import Data

from models import PDGCNConfig
from training.graph_utils import (
    graph_explicit_source_delta,
    graph_physical_source_delta,
    graph_residual_source_delta,
)
from training.tbptt import rollout_window


class RecordingZeroModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.recorded_x = []

    def forward(self, graph):
        self.recorded_x.append(graph.x.detach().clone())
        return graph.x.new_zeros(graph.num_nodes, 1)


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
        self.assertTrue(torch.allclose(graph_physical_source_delta(graph, config), graph.q_surface_star))
        self.assertTrue(torch.allclose(graph_residual_source_delta(graph, config), graph.q_surface_star))

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
        self.assertTrue(torch.equal(graph_residual_source_delta(graph, config), torch.zeros(3, 1)))

    def test_disabled_rollout_keeps_source_features_without_preheating_state(self):
        q_surface = torch.tensor([[0.0], [0.5], [1.0]])
        node_features = torch.zeros(3, 9)
        node_features[:, 7:8] = q_surface
        graph = Data(
            x=node_features,
            q_surface_star=q_surface,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 7)),
            global_attr=torch.tensor([0.0]),
            upwind_nodes=torch.empty(0, dtype=torch.long),
            side_nodes=torch.empty(0, dtype=torch.long),
            downwind_nodes=torch.empty(0, dtype=torch.long),
        )
        graph.num_nodes = 3
        graph.q_feature_index = 7
        graph.delta_t_source_feature_index = 8
        config = PDGCNConfig(
            include_q_in_features=True,
            include_delta_t_source_in_features=True,
            use_explicit_heat_source=False,
            source_coefficient=2.0,
            dt_star=0.5,
        )
        model = RecordingZeroModel(config)

        predictions, _ = rollout_window(model, [graph], torch.zeros(3, 1))

        model_input = model.recorded_x[0]
        self.assertTrue(torch.equal(predictions, torch.zeros(1, 3, 1)))
        self.assertTrue(torch.equal(model_input[:, 6:7], torch.zeros(3, 1)))
        self.assertTrue(torch.allclose(model_input[:, 7:8], q_surface))
        self.assertTrue(torch.allclose(model_input[:, 8:9], q_surface))

    def test_switch_requires_boolean_value(self):
        with self.assertRaisesRegex(ValueError, "use_explicit_heat_source must be a boolean"):
            PDGCNConfig(use_explicit_heat_source=0)


if __name__ == "__main__":
    unittest.main()
