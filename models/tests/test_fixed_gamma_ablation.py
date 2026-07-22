import unittest

import torch

from models import PDGCNConfig
from models.processor.edge_block import EdgeBlock


class FixedGammaAblationTests(unittest.TestCase):
    def test_gamma_is_checkpointed_but_not_trainable(self):
        block = EdgeBlock(PDGCNConfig(hidden_size=8, gamma_upwind=0.8))

        self.assertNotIn("gamma_upwind", dict(block.named_parameters()))
        self.assertIn("gamma_upwind", dict(block.named_buffers()))
        self.assertAlmostEqual(float(block.gamma_upwind), 0.8, places=6)

    def test_upwind_gate_uses_the_fixed_gamma_value(self):
        block = EdgeBlock(PDGCNConfig(hidden_size=8, gamma_upwind=0.8))
        raw_edge_attr = torch.zeros((3, 7), dtype=torch.float32)
        raw_edge_attr[:, 4] = torch.tensor([-1.0, 0.0, 1.0])

        alpha = block._upwind_gate(raw_edge_attr)

        torch.testing.assert_close(alpha[:, 0], torch.tensor([0.2, 1.0, 1.8]))


if __name__ == "__main__":
    unittest.main()
