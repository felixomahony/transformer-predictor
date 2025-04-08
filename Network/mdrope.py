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


        self.time_period_exponents = torch.arange(0, head_dim // 4)
        self.time_periods = torch.pow(geometric_base, 4 * self.time_period_exponents / head_dim)
        
        self.grid_rng = [torch.arange(0, s) for s in grid_shape]
        self.grid_idxes = torch.meshgrid(self.grid_rng)

        self.phases = [gi.flatten()[:, None] / self.time_periods[None, :] for gi in self.grid_idxes]
        self.cos_term = [torch.cos(p) for p in self.phases]
        self.sin_term = [torch.sin(p) for p in self.phases]
        self.cos_term = torch.cat(self.cos_term, dim=-1)[None, None]
        self.sin_term = torch.cat(self.sin_term, dim=-1)[None, None]

        # make the cos and sin term available to move to cuda
        self.cos_term = nn.Parameter(self.cos_term, requires_grad=False)
        self.sin_term = nn.Parameter(self.sin_term, requires_grad=False)

    def forward(self, x):
        """
        Forward pass for MDRoPE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, H, N, L]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, H, N, L]
        """
        assert math.prod(self.grid_shape) == x.shape[-2]

        b, h, n, l = x.shape
        c1 = self.cos_term * x[..., :l//2]
        s1 = self.sin_term * x[..., :l//2]
        c2 = self.cos_term * x[..., l//2:]
        s2 = self.sin_term * x[..., l//2:]
        x = torch.cat([c1 - s2, s1 + c2], dim=-1)

        return x
