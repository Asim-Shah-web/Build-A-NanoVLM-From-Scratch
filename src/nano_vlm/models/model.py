import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class NanoVLM(nn.Module):
    """
    Complete NanoVLM containing:

    Image Encoder
    +
    Text Encoder
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        attention_heads,
        context_window,
        projection_dim=64,
        padding_idx=None,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            projection_dim=projection_dim,
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            max_length=context_window,
            projection_dim=projection_dim,
            padding_idx=padding_idx,
        )

    def encode_image(self, images):

        embeddings = self.image_encoder(
            images
        )

        return F.normalize(
            embeddings,
            dim=-1,
        )

    def encode_text(self, tokens):

        embeddings = self.text_encoder(
            tokens
        )

        return F.normalize(
            embeddings,
            dim=-1,
        )

    def forward(
        self,
        images,
        tokens,
    ):

        image_embeddings = self.encode_image(
            images
        )

        text_embeddings = self.encode_text(
            tokens
        )

        return (
            image_embeddings,
            text_embeddings,
        )