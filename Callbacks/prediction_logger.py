import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np

from sklearn.decomposition import PCA


class PredictionLogger(pl.Callback):
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        plt.close("all")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:

            fig = self._generate_figure(
                outputs, pl_module.args.vit.mask_value, pl_module.args.vit.empty_value
            )

            trainer.logger.experiment.add_figure(
                "Prediction",
                fig,
                global_step=trainer.global_step,
            )

            plt.close(fig)

    def _generate_figure(self, outputs, mask_token, empty_token):
        # Your expensive figure generation code goes here
        # This is just a placeholder example
        fig, ax = plt.subplots(
            1, 3, figsize=(9, 3 * torch.numel(outputs["input_code"][0]) // 64)
        )

        assert "input_code" in outputs
        assert "pred_code" in outputs
        assert "masked_code" in outputs

        assert (
            empty_token == mask_token + 1
        ), "mask_token should be empty_token - 1 for this implementation"

        rainbow_cmap = plt.cm.rainbow
        colors = np.vstack(
            (
                rainbow_cmap(np.linspace(0, 1, mask_token)),
                np.array([[0, 0, 0, 1]]),
                np.array([[1, 1, 1, 1]]),
            )
        )
        custom_cmap = mcolors.ListedColormap(colors)

        ax_input = ax[0]
        ax_input.imshow(
            outputs["input_code"][0].view(-1, 8).detach().cpu().numpy(),
            cmap=custom_cmap,
            vmin=0,
            vmax=empty_token,
        )

        ax_mask = ax[1]
        ax_mask.imshow(
            outputs["masked_code"][0].view(-1, 8).detach().cpu().numpy(),
            cmap=custom_cmap,
            vmin=0,
            vmax=empty_token,
        )

        ax_pred = ax[2]
        pred = (
            torch.argmax(outputs["pred_code"][0], dim=1)
            .view(-1, 8)
            .detach()
            .cpu()
            .numpy()
        )
        ax_pred.imshow(
            pred,
            cmap=custom_cmap,
            vmin=0,
            vmax=empty_token,
        )

        ax_input.set_xticks([])
        ax_input.set_yticks([])
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

        ax_input.set_title("Input Code")
        ax_mask.set_title("Masked Code")
        ax_pred.set_title("Predicted Code")

        plt.tight_layout()

        return fig
