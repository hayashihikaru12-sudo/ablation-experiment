import unittest

import torch

from training.config import TrainConfig
from training.lr_scheduler import build_lr_scheduler, optimizer_lr


class LRSchedulerTests(unittest.TestCase):
    def test_train_config_rejects_invalid_lr_scheduler_settings(self):
        with self.assertRaisesRegex(ValueError, "lr_scheduler"):
            TrainConfig(lr_scheduler="bad", tbptt_window=1)
        with self.assertRaisesRegex(ValueError, "min_lr"):
            TrainConfig(lr=0.1, min_lr=0.2, tbptt_window=1)
        with self.assertRaisesRegex(ValueError, "lr_warmup_epochs"):
            TrainConfig(lr_warmup_epochs=-1, tbptt_window=1)
        with self.assertRaisesRegex(ValueError, "lr_patience"):
            TrainConfig(lr_patience=0, tbptt_window=1)
        with self.assertRaisesRegex(ValueError, "lr_factor"):
            TrainConfig(lr_factor=1.0, tbptt_window=1)
        with self.assertRaisesRegex(ValueError, "seed"):
            TrainConfig(seed=-1, tbptt_window=1)

    def test_warmup_cosine_sets_expected_epoch_lrs(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        scheduler = build_lr_scheduler(
            optimizer,
            TrainConfig(
                lr=0.1,
                epochs=4,
                tbptt_window=1,
                lr_scheduler="warmup_cosine",
                lr_warmup_epochs=2,
                min_lr=0.01,
            ),
        )

        lrs = []
        for epoch in range(4):
            scheduler.begin_epoch(epoch)
            lrs.append(optimizer_lr(optimizer))

        self.assertAlmostEqual(lrs[0], 0.055)
        self.assertAlmostEqual(lrs[1], 0.1)
        self.assertAlmostEqual(lrs[2], 0.055)
        self.assertAlmostEqual(lrs[3], 0.01)

    def test_plateau_reduces_lr_after_patience(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        scheduler = build_lr_scheduler(
            optimizer,
            TrainConfig(
                lr=0.1,
                epochs=4,
                tbptt_window=1,
                lr_scheduler="plateau",
                min_lr=0.02,
                lr_patience=2,
                lr_factor=0.5,
            ),
        )

        scheduler.end_epoch(1.0)
        scheduler.end_epoch(1.1)
        self.assertAlmostEqual(optimizer_lr(optimizer), 0.1)
        scheduler.end_epoch(1.2)
        self.assertAlmostEqual(optimizer_lr(optimizer), 0.05)
        scheduler.end_epoch(1.3)
        scheduler.end_epoch(1.4)
        self.assertAlmostEqual(optimizer_lr(optimizer), 0.025)
        scheduler.end_epoch(1.5)
        scheduler.end_epoch(1.6)
        self.assertAlmostEqual(optimizer_lr(optimizer), 0.02)


if __name__ == "__main__":
    unittest.main()
