import os
import time
import math

import numpy as np

from tqdm import tqdm

# import pca from sklearn
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from Network.transformer import MaskTransformer
from Data import GS_Dataset


class MaskGIT(pl.LightningModule):

    def __init__(self, args):
        """Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__()
        self.args = args

        self.tokens_per_sample = GS_Dataset.datum_size(**args.data.ka)
        self.vit = self.get_network(
            tokens_per_sample=self.tokens_per_sample,
            codebook_size=self.args.vqvae.codebook_n,
            code_dim=self.args.vqvae.hidden_dim,
            **self.args.vit.ka,
        )  # Load Masked Bidirectional Transformer
        self.criterion = nn.CrossEntropyLoss()

        self.sched_func = {
            "root": lambda r: 1 - (r**0.5),
            "linear": lambda r: 1 - r,
            "square": lambda r: 1 - (r**2),
            "cosine": lambda r: torch.cos(r * math.pi * 0.5),
            "arccos": lambda r: torch.arccos(r) / (math.pi * 0.5),
            "none": lambda r: torch.zeros_like(r),
        }

    def get_network(
        self,
        mask_value=1000,
        empty_value=1001,
        r_temp=4.5,
        sm_temp=1.0,
        sched_mode="arccos",
        **kwargs,
    ):
        """return the network, load checkpoint if self.args.resume == True
        :param
            archi -> str: vit|autoencoder, the architecture to load
        :return
            model -> nn.Module: the network
        """
        model = MaskTransformer(
            **kwargs,
        )

        if self.args.vqvae.codebook_path != "":
            model.load_codebook(self.args.vqvae.codebook_path)
        else:
            raise ValueError(
                "Please provide the path to the codebook file in the configuration."
            )

        return model

    def get_mask_code(self, code, mode="arccos", value=None, codebook_size=256):
        """Replace the code token by *value* according the the *mode* scheduler
        :param
         code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
         mode  -> str:                the rate of value to mask
         value -> int:                mask the code by the value
        :return
         masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
         mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        val_to_mask = self.sched_func[mode](r)

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        extended_shape = [code.size(0)] + [1] * (len(code.size()) - 1)
        mask = torch.rand(size=code.size()) < val_to_mask.view(*extended_shape)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """Create a sampling scheduler
        :param
         step  -> int:  number of prediction during inference
         mode  -> str:  the rate of value to unmask
         leave -> bool: tqdm arg on either to keep the bar or not
        :return
         scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        val_to_mask = self.sched_func[mode](r)

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.tokens_per_sample)
        sche = sche.round()
        sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
        sche[-1] += (self.tokens_per_sample) - sche.sum()  # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def calc_loss(self, pred, code):
        """Calculate the loss between the predicted and target code
        :param
         pred -> torch.LongTensor(): bsize * 16 * 16, the predicted code
         code -> torch.LongTensor(): bsize * 16 * 16, the target code
        :return
         loss -> float: the loss value
        """
        loss_dict = {}
        loss = 0

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            pred.reshape(-1, self.args.vqvae.codebook_n + 2), code.view(-1)
        )
        loss_dict["ce_loss"] = ce_loss.item()
        if self.args.learning.lambda_ce > 0:
            loss += ce_loss * self.args.learning.lambda_ce

        # filled loss
        pred_reshaped = pred.reshape(-1, self.args.vqvae.codebook_n + 2)
        code_reshaped = code.view(-1)
        empty_mask = code_reshaped != self.args.vit.empty_value
        filled_loss = F.cross_entropy(
            pred_reshaped[empty_mask],
            code_reshaped[empty_mask],
            ignore_index=self.args.vit.empty_value,
        )
        loss_dict["filled_loss"] = filled_loss.item()
        if self.args.learning.lambda_filled > 0:
            loss += filled_loss * self.args.learning.lambda_filled

        # emptiness loss
        code_empty = (
            torch.maximum(
                code,
                torch.tensor([self.args.vqvae.codebook_n - 1], device=code.device),
            )
            - self.args.vqvae.codebook_n
            + 1
        )

        pred_probs = torch.softmax(pred, dim=-1)
        pred_empty = pred_probs[..., -2:]
        pred_filled = torch.sum(pred_probs[:, :, :-2], dim=-1, keepdim=True)
        pred_e_f = torch.cat([pred_filled, pred_empty], dim=-1)

        logits_e_f = torch.log(pred_e_f + 1e-5)

        emptiness_loss = F.cross_entropy(
            logits_e_f.reshape(-1, 3),
            code_empty.view(-1),
        )
        loss_dict["emptiness_loss"] = emptiness_loss.item()
        if self.args.learning.lambda_empty > 0:
            loss += emptiness_loss * self.args.learning.lambda_empty

        loss_dict["loss_total"] = loss.item()
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # we don't care about the spatial aspects of the code
        code = batch.reshape(batch.size(0), -1)

        # Mask the encoded tokens
        masked_code, _ = self.get_mask_code(
            code,
            mode=self.args.learning.sched_mode_learning,
            value=self.args.vit.mask_value,
            codebook_size=self.args.vqvae.codebook_n,
        )

        with torch.amp.autocast("cuda"):  # half precision
            pred = self.vit(masked_code)
            # Cross-entropy loss
            loss, loss_dict = self.calc_loss(pred, code)

        self.log_all(train=True, loss_dict=loss_dict)
        return {
            "loss": loss,
            "input_code": code,
            "masked_code": masked_code,
            "pred_code": pred,
        }

    def validation_step(self, batch, batch_idx):
        # we don't care about the spatial aspects of the code
        code = batch.reshape(batch.size(0), -1)

        # Mask the encoded tokens
        masked_code, _ = self.get_mask_code(
            code,
            mode=self.args.learning.sched_mode_learning,
            value=self.args.vit.mask_value,
            codebook_size=self.args.vqvae.codebook_n,
        )

        with torch.amp.autocast("cuda"):
            pred = self.vit(masked_code)
            # Cross-entropy loss
            loss, loss_dict = self.calc_loss(pred, code)

        self.log_all(train=False, loss_dict=loss_dict)
        return {
            "loss": loss,
            "input_code": code,
            "masked_code": masked_code,
            "pred_code": pred,
        }

    def log_all(self, train, loss_dict):
        prefix = "train" if train else "val"
        for k, v in loss_dict.items():
            self.log(f"{prefix}/{k}", v, prog_bar=k == "loss_total", logger=True)

    def sample(
        self,
        init_code=None,
        nb_sample=50,
        sm_temp=1,
        w=3,
        randomize="linear",
        r_temp=4.5,
        sched_mode="arccos",
        step=12,
        with_replacement=False,
    ):
        """Generate sample with the MaskGIT model
        :param
         init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
         nb_sample   -> int:              the number of image to generated
         sm_temp     -> float:            the temperature before softmax
         w           -> float:            scale for the classifier free guidance
         randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
         r_temp      -> float:            temperature for the randomness
         sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
         step:       -> int:              number of step for the decoding
        :return
         x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
         code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []  # Save the intermediate masks
        with torch.no_grad():
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (
                    (init_code == self.args.vqvae.codebook_n)
                    .float()
                    .view(nb_sample, self.tokens_per_sample)
                )
            else:  # Initialize a code
                if self.args.vit.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(
                        0,
                        self.args.vqvae.codebook_n,
                        (nb_sample, self.tokens_per_sample),
                    ).to(self.args.run.device)
                else:  # Code initialize with masked tokens
                    code = torch.full(
                        (nb_sample, self.tokens_per_sample),
                        self.args.vit.mask_value,
                    ).to(self.args.run.device)
                mask = torch.ones(nb_sample, self.tokens_per_sample).to(
                    self.args.run.device
                )

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            t_cumulative = 0
            for indice, t in enumerate(scheduler):
                t_cumulative += t
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())
                    t_cumulative = t

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision
                    logit = self.vit(code.clone())

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(
                    prob,
                    2,
                    pred_code.view(
                        nb_sample,
                        self.tokens_per_sample,
                        1,
                    ),
                )

                if (
                    randomize == "linear"
                ):  # add gumbel noise decreasing over the sampling process
                    ratio = indice / (len(scheduler) - 1)
                    rand = (
                        r_temp
                        * np.random.gumbel(
                            size=(
                                nb_sample,
                                self.tokens_per_sample,
                            )
                        )
                        * (1 - ratio)
                    )
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(
                        self.args.run.device
                    )
                elif (
                    randomize == "warm_up"
                ):  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":  # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens if not with replacement
                if not with_replacement:
                    conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                t_ = t if not with_replacement else t_cumulative
                tresh_conf, indice_mask = torch.topk(
                    conf.view(nb_sample, -1), k=t_, dim=-1
                )
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(
                    nb_sample, self.tokens_per_sample
                )
                if not with_replacement:
                    f_mask = (
                        mask.view(nb_sample, self.tokens_per_sample).float()
                        * conf.view(nb_sample, self.tokens_per_sample).float()
                    ).bool()
                else:
                    f_mask = torch.ones_like(code).bool()
                code[f_mask] = pred_code.view(nb_sample, self.tokens_per_sample)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(
                    pred_code.view(nb_sample, self.tokens_per_sample).clone()
                )
                l_mask.append(mask.view(nb_sample, self.tokens_per_sample).clone())

            # decode the final prediction
            # _code = torch.clamp(code, 0, self.codebook_size + 1)
            # x = self.ae.decode_code(_code)

        self.vit.train()
        return code, l_codes, l_mask

    def configure_optimizers(self):

        params = self.vit.parameters()
        lr = self.args.learning.lr

        optimizer = optim.AdamW(params, lr, weight_decay=1e-5, betas=(0.9, 0.96))

        return optimizer
