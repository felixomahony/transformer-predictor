import torch
import torch.nn as nn
import math
import numpy as np

from Network.mdrope import MDRoPE


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        rope=False,
        grid_shape=None,
        geometric_base=100,
        dropout=0.0,
        window_size=None,
        window_attention=False,
    ):
        """
        Custom Multi-Head Attention implementation.

        Args:
            embed_dim (int): Dimensionality of the input embeddings
            num_heads (int): Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        # Ensure embed_dim is divisible by num_heads
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.head_dim)

        self.use_rope = rope
        if self.use_rope:
            self.rope = MDRoPE(self.head_dim, grid_shape, geometric_base)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        if window_attention and window_size is not None:
            indices = torch.tensor(np.indices(grid_shape)).flatten(1).T
            distances = (
                torch.abs(indices[:, None, :] - indices[None, :, :]) * 2
            )  # 2 because the window size is applied in both directions
            window = torch.tensor(window_size)
            self.attention_mask = torch.all(distances < window[None, None, :], dim=-1)
        else:
            self.attention_mask = None

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, L]
            mask (torch.Tensor, optional): Attention mask of shape [B, 1, N, N]

        Returns:
            torch.Tensor: Output tensor of shape [B, N, L]
        """
        # Input shape: [B, N, L]
        batch_size, seq_len, embed_dim = x_q.size()

        # Project inputs
        query = self.query_proj(x_q)  # [B, N, L]
        key = self.key_proj(x_k)  # [B, N, L]
        value = self.value_proj(x_v)  # [B, N, L]

        # Reshape and split into multiple heads
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if self.use_rope:
            query = self.rope(query)
            key = self.rope(key)

        if self.attention_mask is None:
            # Compute attention scores
            # [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
            attention_scores = (query @ key.transpose(-2, -1)) / self.scale
            # Apply mask if provided
            if mask is not None:
                attention_scores = attention_scores.masked_fill(
                    mask == 0, torch.finfo(torch.float16).min
                )
            # cut out any inf values
            attention_scores = attention_scores.masked_fill(
                attention_scores == float("-inf"), torch.finfo(torch.float16).min
            )
            attention_scores = attention_scores.masked_fill(
                attention_scores == float("inf"), torch.finfo(torch.float16).max
            )
            # Softmax attention scores
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            # Apply attention to values
            # [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
            context = attention_probs @ value
        else:
            # we assume that we are using the mask to reduce memory, so computing the full matrix is inefficient
            context_lst = []
            kt = key.transpose(-2, -1)
            for row in range(query.shape[-2]):
                row_mask = self.attention_mask[row]

                # [B, H, 1, N_m]
                attention_scores = query[..., row : row + 1, :] @ kt[..., row_mask]
                # cut out any inf values
                attention_scores = attention_scores.masked_fill(
                    attention_scores == float("-inf"), torch.finfo(torch.float16).min
                )
                attention_scores = attention_scores.masked_fill(
                    attention_scores == float("inf"), torch.finfo(torch.float16).max
                )
                attention_probs = torch.softmax(attention_scores, dim=-1)
                attention_probs = self.attention_dropout(attention_probs)

                # [B, H, 1, N_m] @ [B, H, N_m, D] -> [B, H, 1, D]
                context_elem = attention_probs @ value[..., row_mask, :]

                context_lst.append(context_elem)

            # [B, H, N, D]
            context = torch.cat(context_lst, dim=-2)

        # Reshape and combine heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        # Final projection
        output = self.output_proj(context)
        output = self.output_dropout(output)

        # return the attention value and the attention weights
        return (output, attention_probs)


# Example usage
def test_multi_head_attention():
    # Hyperparameters
    batch_size = 2
    num_tokens = 49
    embed_dim = 64
    num_heads = 8

    # Create random input tensor
    x = torch.randn(batch_size, num_tokens, embed_dim)

    # Initialize multi-head attention
    mha = MultiHeadAttention(
        embed_dim, num_heads, rope=True, grid_shape=(7, 7), geometric_base=100
    )

    # Optional: Create a mask (1 for allowed, 0 for masked)
    mask = torch.ones(batch_size, 1, num_tokens, num_tokens)

    # Forward pass
    output = mha(x, x, x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
