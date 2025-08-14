import torch
from torch import nn
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
        x = torch.clamp(
            x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
        )
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
    def __init__(
        self,
        embed_dim,
        num_heads,
        grid_shape,
        dropout=0.0,
        rope=False,
        split_spatio_temporal=False,
        split_spatio=False,
        window_attention=False,
        window_size=None,
    ):
        """Initialize the Attention module.
        :param:
            embed_dim     -> int : Dimension of the embedding
            num_heads     -> int : Number of heads
            dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.split_spatio = split_spatio
        self.split_spatio_temporal = split_spatio_temporal
        self.grid_shape = grid_shape

        if split_spatio:
            self.mha_split = torch.nn.ModuleList(
                [
                    MultiHeadAttention(
                        embed_dim,
                        num_heads,
                        rope=rope,
                        grid_shape=grid_shape[i : i + 1],
                        geometric_base=100,
                        dropout=dropout,
                        window_attention=window_attention,
                        window_size=window_size,
                    )
                    for i in range(len(grid_shape))
                ]
            )
        elif split_spatio_temporal:
            assert len(grid_shape) == 4
            self.mha_spatio = MultiHeadAttention(
                embed_dim,
                num_heads,
                rope=rope,
                grid_shape=grid_shape[-3:],
                geometric_base=100,
                dropout=dropout,
                window_attention=window_attention,
                window_size=window_size,
            )
            self.mha_temporal = MultiHeadAttention(
                embed_dim,
                num_heads,
                rope=rope,
                grid_shape=[grid_shape[0]],
                geometric_base=100,
                dropout=dropout,
                window_attention=window_attention,
                window_size=window_size,
            )
        else:
            self.mha = MultiHeadAttention(
                embed_dim,
                num_heads,
                rope=rope,
                grid_shape=grid_shape,
                geometric_base=100,
                dropout=dropout,
                window_attention=window_attention,
                window_size=window_size,
            )

    def forward(self, x):
        """Forward pass through the Attention module.
        :param:
            x -> torch.Tensor: Input tensor
        :return
            attention_value  -> torch.Tensor: Output the value of the attention
            attention_weight -> torch.Tensor: Output the weight of the attention
        """
        if self.split_spatio:
            b, s, f = x.shape
            x = x.view(b, *self.grid_shape, f)
            for i in range(len(self.grid_shape)):
                x = torch.transpose(x, i + 1, -2)
                shape = x.shape
                x = x.reshape(-1, self.grid_shape[i], f)
                x, attention_weight = self.mha_split[i](x, x, x)
                x = x.reshape(*shape)
                x = torch.transpose(x, i + 1, -2)
            attention_value = x.reshape(b, s, f)
        elif self.split_spatio_temporal:
            b = x.shape[0]
            t = self.grid_shape[0]
            s = x.shape[1] // t
            f = x.shape[2]

            x_spatial = x.view(
                b * t,
                s,
                f,
            )
            a_v, a_w = self.mha_spatio(x_spatial, x_spatial, x_spatial)
            f = a_v.shape[2]

            x_temporal = (
                a_v.view(
                    b,
                    t,
                    s,
                    f,
                )
                .permute(0, 2, 1, 3)
                .reshape(
                    b * s,
                    t,
                    f,
                )
            )
            attention_value, attention_weight = self.mha_temporal(
                x_temporal, x_temporal, x_temporal
            )
            attention_value = (
                attention_value.reshape(b, s, t, -1)
                .permute(0, 2, 1, 3)
                .reshape(b, t * s, -1)
            )
        else:
            attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        grid_shape,
        dropout=0.0,
        rope=False,
        split_spatio_temporal=False,
        split_spatio=False,
        window_attention=False,
        window_size=None,
    ):
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
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads,
                                grid_shape,
                                dropout=dropout,
                                rope=rope,
                                split_spatio_temporal=split_spatio_temporal,
                                split_spatio=split_spatio,
                                window_attention=window_attention,
                                window_size=window_size,
                            ),
                        ),
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
            x = torch.clamp(
                x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
            )
            x = ff(x) + x
            x = torch.clamp(
                x, torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
            )
            l_attn.append(attention_weight)
        return x, l_attn
