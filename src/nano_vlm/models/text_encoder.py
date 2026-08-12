import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Small Transformer-based text encoder for NanoVLM.

    Input:
        [B, L]

    Output:
        [B, projection_dim]
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        attention_heads,
        max_length,
        projection_dim=64,
        padding_idx=None,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # ----------------------------------------------------
        # Learnable positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                max_length,
                embed_dim,
            ) * 0.02
        )

        # ----------------------------------------------------
        # Multi-head self-attention
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        self.layer_norm = nn.LayerNorm(
            embed_dim
        )

        # ----------------------------------------------------
        # Projection into shared embedding space
        # ----------------------------------------------------

        self.projection = nn.Linear(
            embed_dim,
            projection_dim,
        )

        self.max_length = max_length

    def forward(self, tokens):

        # ----------------------------------------------------
        # tokens:
        # [B, L]
        # ----------------------------------------------------

        batch_size, sequence_length = tokens.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds max_length {self.max_length}."
            )

        # ----------------------------------------------------
        # Token embeddings
        # [B, L, D]
        # ----------------------------------------------------

        x = self.token_embedding(tokens)

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------

        positions = self.position_embedding[
            :,
            :sequence_length,
            :
        ]

        x = x + positions

        # ----------------------------------------------------
        # Self-attention
        # [B, L, D]
        # ----------------------------------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x,
        )

        # ----------------------------------------------------
        # Residual connection + LayerNorm
        # ----------------------------------------------------

        x = self.layer_norm(
            x + attention_output
        )

        # ----------------------------------------------------
        # Mean pooling over sequence
        # [B, D]
        # ----------------------------------------------------

        x = x.mean(dim=1)

        # ----------------------------------------------------
        # Projection
        # [B, projection_dim]
        # ----------------------------------------------------

        x = self.projection(x)

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        x = F.normalize(
            x,
            dim=-1,
        )

        return x