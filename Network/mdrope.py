import torch
import torch.nn as nn
import math


class MDRoPE(nn.Module):
    def __init__(self, head_dim, grid_shape: tuple, geometric_base=100):
        """
        Multi-Dimensional Representation of Positional Encoding (MDRoPE) layer.

        Args:
            embed_dim (int): Dimensionality of the input embeddings
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(MDRoPE, self).__init__()

        self.head_dim = head_dim
        self.grid_shape = grid_shape
        self.geometric_base = geometric_base
        self.n_dims = len(grid_shape)

        self.time_period_exponents = torch.arange(
            0, math.ceil(head_dim / 2 / self.n_dims)
        )
        self.time_periods = torch.pow(
            geometric_base, 2 * self.n_dims * self.time_period_exponents / head_dim
        )

        self.grid_rng = [torch.arange(0, s) for s in grid_shape]
        self.grid_idxes = torch.meshgrid(self.grid_rng)

        self.phases = [
            gi.flatten()[:, None] / self.time_periods[None, :] for gi in self.grid_idxes
        ]
        cos_term = [torch.cos(p) for p in self.phases]
        sin_term = [torch.sin(p) for p in self.phases]
        cos_term = torch.cat(cos_term, dim=-1)[None, None]
        sin_term = torch.cat(sin_term, dim=-1)[None, None]

        # make the cos and sin term available to move to cuda
        self.register_buffer("cos_term", cos_term)
        self.register_buffer("sin_term", sin_term)

    def forward(self, x):
        """
        Forward pass for MDRoPE layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, H, N, L]

        Returns:
            torch.Tensor: Output tensor of shape [B, H, N, L]
        """
        assert (
            math.prod(self.grid_shape) == x.shape[-2]
        ), f"The product of grid shape dimensions ({self.grid_shape}) must equal the number of elements in the tensor ({x.shape[-2]})"

        b, h, n, l = x.shape
        c1 = self.cos_term * x[..., : l // 2]
        s1 = self.sin_term * x[..., : l // 2]
        c2 = self.cos_term * x[..., l // 2 :]
        s2 = self.sin_term * x[..., l // 2 :]
        x = torch.cat([c1 - s2, s1 + c2], dim=-1)

        return x
