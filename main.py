# Main file to launch training or evaluation
import os
import random
import numpy as np
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from Trainer.maskgit import MaskGIT
from arguments.args import Args
from Data import GS_DataModule
from Callbacks.codebook_logger import CodebookLogger
from Callbacks.prediction_logger import PredictionLogger


def main(args):
    a = Args("./arguments/default_args.yaml", args)

    if a.run.seed > 0:  # Set the seed for reproducibility
        seed = a.run.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    maskgit = MaskGIT(a)
    data_module = GS_DataModule(**a.data.ka)
    if a.run.test_only:  # Evaluate the networks
        maskgit.eval()
    else:  # Begin training
        logger = TensorBoardLogger(**a.logging.ka)
        trainer = Trainer(
            **a.trainer.ka,
            logger=logger,
            callbacks=[
                CodebookLogger(log_every_n_steps=a.trainer.log_every_n_steps),
                PredictionLogger(log_every_n_steps=a.trainer.log_every_n_steps),
            ],
        )
        trainer.fit(maskgit, data_module)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
