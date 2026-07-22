import unittest

import torch

from models import PDGCNConfig
from models.processor.edge_block import EdgeBlock


class UpwindGateOneAblationTests(unittest.TestCase):
    def test_branch_default_disables_upwind_gate(self):
        self.assertFalse(PDGCNConfig().use_upwind_gate)

    def test_disabled_upwind_gate_is_one_for_every_edge(self):
        block = EdgeBlock(PDGCNConfig(hidden_size=8, gamma_upwind=0.8))
        raw_edge_attr = torch.randn((5, 7), dtype=torch.float32)
        raw_edge_attr[:, 4] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

        alpha = block._upwind_gate(raw_edge_attr)

        torch.testing.assert_close(alpha, torch.ones((5, 1)))
        self.assertIn("gamma_upwind", dict(block.named_parameters()))

    def test_use_upwind_gate_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "use_upwind_gate"):
            PDGCNConfig(use_upwind_gate=1)


if __name__ == "__main__":
    unittest.main()
