import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA


class StraightThroughAccuracy(pl.Callback):
    def __init__(self):
        super().__init__()

    def tally_correct(self, pl_module, outputs):
        masked_code = outputs["masked_code"]
        pred_code = torch.argmax(outputs["pred_code"], dim=-1)

        mask_token_mask = masked_code == pl_module.args.vit.mask_value

        masked_code = masked_code[~mask_token_mask]
        pred_code = pred_code[~mask_token_mask]
        n_tokens = masked_code.numel()
        n_tokens_correct = (masked_code == pred_code).sum().item()

        empty_mask = masked_code == pl_module.args.vit.empty_value
        masked_code = masked_code[~empty_mask]
        pred_code = pred_code[~empty_mask]
        non_empty_tokens = masked_code.numel()
        non_empty_tokens_correct = (masked_code == pred_code).sum().item()

        return n_tokens, n_tokens_correct, non_empty_tokens, non_empty_tokens_correct

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_n_tokens = 0
        self.train_n_tokens_correct = 0
        self.train_non_empty_tokens = 0
        self.train_non_empty_tokens_correct = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        (
            n_tokens,
            n_tokens_correct,
            non_empty_tokens,
            non_empty_tokens_correct,
        ) = self.tally_correct(pl_module, outputs)
        self.train_n_tokens += n_tokens
        self.train_n_tokens_correct += n_tokens_correct
        self.train_non_empty_tokens += non_empty_tokens
        self.train_non_empty_tokens_correct += non_empty_tokens_correct

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            "train/straight_through_accuracy",
            self.train_n_tokens_correct / self.train_n_tokens,
        )
        pl_module.log(
            "train/non_empty_straight_through_accuracy",
            self.train_non_empty_tokens_correct / self.train_non_empty_tokens,
        )

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_n_tokens = 0
        self.val_n_tokens_correct = 0
        self.val_non_empty_tokens = 0
        self.val_non_empty_tokens_correct = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        (
            n_tokens,
            n_tokens_correct,
            non_empty_tokens,
            non_empty_tokens_correct,
        ) = self.tally_correct(pl_module, outputs)

        self.val_n_tokens += n_tokens
        self.val_n_tokens_correct += n_tokens_correct
        self.val_non_empty_tokens += non_empty_tokens
        self.val_non_empty_tokens_correct += non_empty_tokens_correct

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log(
            "val/straight_through_accuracy",
            self.val_n_tokens_correct / self.val_n_tokens,
        )
        pl_module.log(
            "val/non_empty_straight_through_accuracy",
            self.val_non_empty_tokens_correct / self.val_non_empty_tokens,
        )
