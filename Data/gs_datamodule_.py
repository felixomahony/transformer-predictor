import pytorch_lightning as lightning
import torch

from Data import GS_Dataset

class GS_DataModule(lightning.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = GS_Dataset(train=True, **kwargs)
        self.val_dataset = GS_Dataset(train=False, **kwargs)

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )