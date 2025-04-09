import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA


class PredictionLogger(pl.Callback):
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

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

            plt.close(fig)  # Close the figure to free memory

    def _generate_figure(self, outputs, mask_token, empty_token):
        # Your expensive figure generation code goes here
        # This is just a placeholder example
        fig, ax = plt.subplots(1, 3)

        assert "input_code" in outputs
        assert "pred_code" in outputs
        assert "masked_code" in outputs

        def replace_mask_empty(tokens):
            tokens_8 = tokens % 8
            tokens_8[tokens == mask_token] = 8
            tokens_8[tokens == empty_token] = 9
            return tokens_8

        ax_input = ax[0]
        ax_input.imshow(
            replace_mask_empty(outputs["input_code"][0])
            .view(-1, 7)
            .detach()
            .cpu()
            .numpy(),
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        ax_input.axis("off")

        ax_mask = ax[1]
        ax_mask.imshow(
            replace_mask_empty(outputs["masked_code"][0])
            .view(-1, 7)
            .detach()
            .cpu()
            .numpy(),
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        ax_mask.axis("off")

        ax_pred = ax[2]
        pred = torch.argmax(outputs["pred_code"][0], dim=1)
        ax_pred.imshow(
            replace_mask_empty(pred).view(-1, 7).detach().cpu().numpy(),
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        ax_pred.axis("off")

        ax_input.set_title("Input Code")
        ax_mask.set_title("Masked Code")
        ax_pred.set_title("Predicted Code")

        plt.tight_layout()

        return fig
