# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from Network.attention import MultiHeadAttention

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """PreNorm module to apply layer normalization before a given function
        :param:
            dim  -> int: Dimension of the input
            fn   -> nn.Module: The function to apply after layer normalization
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward pass through the PreNorm module
        :param:
            x        -> torch.Tensor: Input tensor
            **kwargs -> _ : Additional keyword arguments for the function
        :return
            torch.Tensor: Output of the function applied after layer normalization
        """
        x = torch.clamp(x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """Initialize the Multi-Layer Perceptron (MLP).
        :param:
            dim        -> int : Dimension of the input
            dim        -> int : Dimension of the hidden layer
            dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward pass through the MLP module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """Initialize the Attention module.
        :param:
            embed_dim     -> int : Dimension of the embedding
            num_heads     -> int : Number of heads
            dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        # self.mha = nn.MultiheadAttention(
        #     embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True
        # )
        self.mha = MultiHeadAttention(embed_dim, num_heads, rope=True, grid_shape=(7,7), geometric_base=100)

    def forward(self, x):
        """Forward pass through the Attention module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            attention_value  -> torch.Tensor: Output the value of the attention
            attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        """Initialize the Attention module.
        :param:
            dim       -> int : number of hidden dimension of attention
            depth     -> int : number of layer for the transformer
            heads     -> int : Number of heads
            mlp_dim   -> int : number of hidden dimension for mlp
            dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        """Forward pass through the Attention module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            x -> torch.Tensor: Output of the Transformer
            l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = torch.clamp(x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
            x = ff(x) + x
            x = torch.clamp(x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
            l_attn.append(attention_weight)
        return x, l_attn


class MaskTransformer(nn.Module):
    def __init__(
        self,
        img_size=256,
        code_dim=768,
        hidden_dim=768,
        codebook_size=1024,
        depth=24,
        heads=8,
        mlp_dim=3072,
        dropout=0.1,
        dims=3,
        patch_size=15,
        learned_pos_emb=False,
        tokens_per_sample=7**3
        # nclass=1000,
    ):
        """Initialize the Transformer model.
        :param:
            img_size       -> int:     Input image size (default: 256)
            hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
            codebook_size  -> int:     Size of the codebook (default: 1024)
            depth          -> int:     Depth of the transformer (default: 24)
            heads          -> int:     Number of attention heads (default: 8)
            mlp_dim        -> int:     MLP dimension (default: 3072)
            dropout        -> float:   Dropout rate (default: 0.1)
            nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        # self.nclass = nclass
        # self.patch_size = img_size // 16
        self.patch_size = patch_size
        self.tokens_per_sample = tokens_per_sample
        self.codebook_size = codebook_size
        # self.tok_emb = nn.Embedding(
        #     codebook_size + 1 + nclass + 1, hidden_dim
        # )  # +1 for the mask of the viz token, +1 for mask of the class
        # self.tok_emb = nn.Embedding(
        #     codebook_size + 2, code_dim
        # )  # +1 for the mask of the viz token. +1 for empty token
        self.mask_token = nn.Parameter(torch.randn((1, code_dim)))
        self.empty_token = nn.Parameter(torch.randn((1, code_dim)))
        if learned_pos_emb:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.tokens_per_sample, code_dim)
            )
            nn.init.trunc_normal_(self.pos_emb, 0.0, 0.02)
        else:
            pos_enc_dim = code_dim // 2
            pos_enc_rng = torch.linspace(0, 1, steps=pos_enc_dim)[None, :]
            pos_array = torch.arange(self.tokens_per_sample)[:, None].float()
            sin_enc = torch.sin(pos_array / ((self.tokens_per_sample / torch.pi / 2) ** pos_enc_rng))
            cos_enc = torch.cos(pos_array / ((self.tokens_per_sample / torch.pi / 2) ** pos_enc_rng))

            self.pos_emb = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)[None]
            self.pos_emb = torch.zeros_like(self.pos_emb)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)

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
            dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=code_dim),
            nn.GELU(),
            nn.LayerNorm(code_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(
            torch.zeros(
                # (self.patch_size * self.patch_size) + 1, codebook_size + 1 + nclass + 1
                self.tokens_per_sample,
                codebook_size
                + 2,  # +1 for the mask of the viz token. +1 for empty token
            )
        )

    def tok_emb_weight(self):
        """Return the weight of the token embedding.
        :return:
            torch.FloatTensor: Weight of the token embedding
        """
        if self.codebook.device != self.mask_token.device:
            self.codebook = self.codebook.to(self.mask_token.device)
        return torch.cat([self.codebook, self.mask_token, self.empty_token], 0)

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
        self.codebook = torch.from_numpy(codebook).float()
        # self.tok_emb.weight.data[:-2] = codebook  # keep space for mask token
        # # freeze all but the last row of the codebook
        # self.tok_emb.weight.requires_grad = True
        # with torch.no_grad():
        #     self.tok_emb.weight[:-2].requires_grad_(False)

    def forward(self, img_token, y=None, drop_label=None, return_attn=False):
        """Forward.
        :param:
            img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            y              -> torch.LongTensor: condition class to generate
            drop_label     -> torch.BoolTensor: either or not to drop the condition
            return_attn    -> Bool: return the attn for visualization
        :return:
            logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
            attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b = img_token.size(0)

        # cls_token = (
        #     y.view(b, -1) + self.codebook_size + 1
        # )  # Shift the class token by the amount of codebook

        # cls_token[drop_label] = self.codebook_size + 1 + self.nclass  # Drop condition
        # input = torch.cat(
        #     [img_token.view(b, -1), cls_token.view(b, -1)], -1
        # )  # concat visual tokens and class tokens
        input = img_token.view(b, -1)
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = (
            torch.matmul(x, self.tok_emb_weight().T) + self.bias
        )  # Shared layer with the embedding

        if return_attn:  # return list of attention
            return (
                logit[:, : self.tokens_per_sample, : self.codebook_size + 2],
                attn,
            )

        return logit[:, : self.tokens_per_sample, : self.codebook_size + 2]
