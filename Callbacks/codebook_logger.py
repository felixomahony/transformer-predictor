import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


class CodebookLogger(pl.Callback):
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:

            fig = self._generate_figure(pl_module, batch, outputs)

            trainer.logger.experiment.add_figure(
                "Codebook",
                fig,
                global_step=trainer.global_step,
            )

            plt.close(fig)  # Close the figure to free memory

    def _generate_figure(self, pl_module, batch, outputs):
        # Your expensive figure generation code goes here
        # This is just a placeholder example
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        codebook = pl_module.vit.codebook.detach().cpu().numpy()
        codebook_pca = PCA(n_components=2)
        codebook_pca.fit(codebook)

        codebook_2d = codebook_pca.transform(codebook)
        ax.scatter(
            codebook_2d[:, 0],
            codebook_2d[:, 1],
            color="gray",
            s=10,
        )

        mask_token_2d = codebook_pca.transform(
            pl_module.vit.mask_token.detach().cpu().numpy()
        )
        ax.scatter(
            mask_token_2d[:, 0],
            mask_token_2d[:, 1],
            c="red",
            label="Mask Token",
            s=30,
        )

        empty_token_2d = codebook_pca.transform(
            pl_module.vit.empty_token.detach().cpu().numpy()
        )
        ax.scatter(
            empty_token_2d[:, 0],
            empty_token_2d[:, 1],
            c="blue",
            label="Empty Token",
            s=30,
        )

        ax.set_title("Codebook")
        ax.axis("off")
        ax.legend()
        return fig
