# Main file to launch training or evaluation
import os
import random
import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from Trainer.maskgit import MaskGIT
from arguments.args import Args
from Data import GS_DataModule
from Callbacks.codebook_logger import CodebookLogger
from Callbacks.prediction_logger import PredictionLogger
from Callbacks.straight_through_accuracy import StraightThroughAccuracy
from Callbacks.accuracy import Accuracy

from flx_utils import FlxArgs


def main(args):
    # TODO: Remove
    a = FlxArgs("./arguments/4d_args.yaml", args)

    pl.seed_everything(a.run.seed)

    if a.run.seed > 0:  # Set the seed for reproducibility
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    if a.run.resume:
        pth = a.run.resume_pth
        if not pth:
            pth = os.path.join(
                a.logging.log_dir,
                a.logging.name,
                a.logging.version,
                "model.ckpt",
            )
        maskgit = MaskGIT.load_from_checkpoint(
            a,
            checkpoint_path=pth,
        )

    else:
        maskgit = MaskGIT(a)
    data_module = GS_DataModule(codebook_size=a.vqvae.codebook_n, **a.data.ka)
    if a.run.test_only:  # Evaluate the networks
        maskgit.eval()
    else:  # Begin training
        logger = TensorBoardLogger(**a.logging.ka)
        trainer = Trainer(
            devices=3,
            sync_batchnorm=True,
            strategy="ddp",
            **a.trainer.ka,
            logger=logger,
            callbacks=[
                ModelCheckpoint(
                    save_top_k=1,
                    verbose=True,
                    monitor=None,  # Not monitoring any metric
                    filename="model",
                    every_n_epochs=a.trainer.max_epochs,
                    enable_version_counter=False,
                ),
                CodebookLogger(log_every_n_steps=a.trainer.log_every_n_steps),
                PredictionLogger(log_every_n_steps=a.trainer.log_every_n_steps),
                StraightThroughAccuracy(),
                Accuracy(),
            ],
        )
        trainer.fit(maskgit, data_module)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
