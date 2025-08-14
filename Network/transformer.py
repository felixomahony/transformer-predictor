# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from Network.base_modules import TransformerEncoder
from enum import Enum


# positional embedding enum
class PositionalEmbedding(Enum):
    learned = 1
    sinusoidal = 2
    rope = 3


class MaskTransformer(nn.Module):
    def __init__(
        self,
        code_dim=768,
        hidden_dim=768,
        codebook_size=1024,
        depth=24,
        heads=8,
        mlp_dim=3072,
        dropout=0.1,
        dims=3,
        positional_embedding="rope",
        tokens_per_sample=7**3,
        grid_shape=(7, 7, 7),
        learnable_codebook=False,
        predict_logits=False,
        normalise_embeddings=False,
        normalise_transformer_output=False,
        remove_final_two_layers=False,
        pass_through_tokens=False,
        split_spatio_temporal=False,
        split_spatio=False,
        window_attention=False,
        window_size=None,
    ):
        """Initialize the Transformer model.
        :param:
            hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
            codebook_size  -> int:     Size of the codebook (default: 1024)
            depth          -> int:     Depth of the transformer (default: 24)
            heads          -> int:     Number of attention heads (default: 8)
            mlp_dim        -> int:     MLP dimension (default: 3072)
            dropout        -> float:   Dropout rate (default: 0.1)
            nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        self.tokens_per_sample = tokens_per_sample
        self.codebook_size = codebook_size
        self.mask_token = nn.Parameter(torch.randn((1, code_dim)))
        self.empty_token = nn.Parameter(torch.randn((1, code_dim)))

        self.learnable_codebook = learnable_codebook
        self.predict_logits = predict_logits
        self.normalise_embeddings = normalise_embeddings
        self.normalise_transformer_output = normalise_transformer_output
        if self.normalise_transformer_output and self.predict_logits:
            raise ValueError(
                "It does not make sense to normalize the transformer output and predict logits at the same time. It equates to arbitrary temperature scaling."
            )
        if self.normalise_transformer_output and not self.normalise_embeddings:
            raise ValueError(
                "If you want to normalize the transformer output, you need to normalize the embeddings as well."
            )
        self.pass_through_tokens = pass_through_tokens

        self.positional_embedding = PositionalEmbedding[positional_embedding]

        if self.positional_embedding == PositionalEmbedding.learned:
            # Learned positional embedding
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.tokens_per_sample, code_dim)
            )
            nn.init.trunc_normal_(self.pos_emb, 0.0, 0.02)
        elif self.positional_embedding == PositionalEmbedding.sinusoidal:
            pos_enc_dim = code_dim // 2
            pos_enc_rng = torch.linspace(0, 1, steps=pos_enc_dim)[None, :]
            pos_array = torch.arange(self.tokens_per_sample)[:, None].float()
            sin_enc = torch.sin(
                pos_array / ((self.tokens_per_sample / torch.pi / 2) ** pos_enc_rng)
            )
            cos_enc = torch.cos(
                pos_array / ((self.tokens_per_sample / torch.pi / 2) ** pos_enc_rng)
            )

            self.pos_emb = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)[None]
            self.pos_emb = nn.Parameter(
                self.pos_emb, requires_grad=False
            )  # Changed to requires_grad=True
        elif self.positional_embedding == PositionalEmbedding.rope:
            # then the positional embedding is done later
            self.pos_emb = 0
            pass
            # RoPE positional embedding
            # self.pos_emb = nn.Parameter(
            #     torch.zeros(1, self.tokens_per_sample, code_dim), requires_grad=False
            # )

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(code_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=code_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            grid_shape=grid_shape,
            dropout=dropout,
            rope=self.positional_embedding == PositionalEmbedding.rope,
            split_spatio_temporal=split_spatio_temporal,
            split_spatio=split_spatio,
            window_attention=window_attention,
            window_size=window_size,
        )

        # Last layer after the Transformer block
        final_dim = codebook_size + 2 if self.predict_logits else code_dim
        last_layer = [
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            # nn.Linear(in_features=hidden_dim, out_features=code_dim),
            nn.Linear(in_features=hidden_dim, out_features=final_dim),
            # nn.GELU(),
            # # nn.LayerNorm(code_dim, eps=1e-12),
            # nn.LayerNorm(final_dim, eps=1e-12),
        ]
        if not remove_final_two_layers:
            last_layer.append(nn.GELU())
            last_layer.append(nn.LayerNorm(final_dim, eps=1e-12))
        self.last_layer = nn.Sequential(*last_layer)

        # Bias for the last linear output
        # self.bias = nn.Parameter(
        #     torch.zeros(
        #         # (self.patch_size * self.patch_size) + 1, codebook_size + 1 + nclass + 1
        #         self.tokens_per_sample,
        #         codebook_size
        #         + 2,  # +1 for the mask of the viz token. +1 for empty token
        #     )
        # )

    def tok_emb_weight(self):
        """Return the weight of the token embedding.
        :return:
            torch.FloatTensor: Weight of the token embedding
        """
        if self.codebook.device != self.mask_token.device:
            self.codebook = self.codebook.to(self.mask_token.device)

        cb = torch.cat([self.codebook, self.mask_token, self.empty_token], 0)
        if self.normalise_embeddings:
            cb = cb / (cb.norm(dim=-1, keepdim=True) + 1e-6)
        return cb

    def tok_emb(self, x):
        """Embedding function for the tokens.
        :param:
            x -> torch.LongTensor: Input tensor
        :return:
            torch.FloatTensor: Embedded tensor
        """
        weight = self.tok_emb_weight()
        return F.embedding(x, weight)

    def load_codebook(self, codebook_path):
        """Load the codebook from the given path.
        :param:
            codebook_path -> str: Path to the codebook
        """
        codebook = np.load(codebook_path)
        codebook = torch.from_numpy(codebook).float()

        if self.learnable_codebook:
            self.codebook = torch.nn.Parameter(
                self.codebook, requires_grad=self.learnable_codebook
            )
        else:
            self.register_buffer("codebook", codebook)

    def forward(self, img_token, return_attn=False):
        """Forward.
        :param:
            img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            return_attn    -> Bool: return the attn for visualization
        :return:
            logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
            attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b = img_token.size(0)

        input = img_token.view(b, -1)
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        if self.pass_through_tokens:
            # x = x + (tok_embeddings - x).detach()
            x = tok_embeddings

        if self.normalise_transformer_output:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)

        if self.predict_logits:
            logit = x
        else:
            output_embed = self.tok_emb_weight().T
            logit = (
                torch.matmul(x, output_embed)
                # + self.bias
            )  # Shared layer with the embedding

        if return_attn:  # return list of attention
            return (
                logit[:, : self.tokens_per_sample, : self.codebook_size + 2],
                attn,
            )

        return logit[:, : self.tokens_per_sample, : self.codebook_size + 2]
