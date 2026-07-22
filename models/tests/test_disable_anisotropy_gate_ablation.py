import unittest

import torch

from models import PDGCNConfig
from models.processor.edge_block import EdgeBlock


class DisableAnisotropyGateAblationTests(unittest.TestCase):
    def test_branch_default_disables_anisotropy_gate(self):
        self.assertFalse(PDGCNConfig().use_aniso_gate)

    def test_disabled_gate_returns_unity_without_changing_parameter_budget(self):
        block = EdgeBlock(PDGCNConfig(hidden_size=8))
        raw_edge_attr = torch.randn((4, 7), dtype=torch.float32)
        message = torch.randn((4, 8), dtype=torch.float32)

        gate = block._aniso_gate(raw_edge_attr, message)

        torch.testing.assert_close(gate, torch.ones_like(message))
        self.assertGreater(sum(p.numel() for p in block.aniso_mlp.parameters()), 0)

    def test_use_aniso_gate_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "use_aniso_gate"):
            PDGCNConfig(use_aniso_gate=1)


if __name__ == "__main__":
    unittest.main()
