"""CLIP semantic conditioning and degradation-aware cross-attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import GatedDconvFeedForward, LayerNorm2d


class OpenCLIPImageEncoder(nn.Module):
    """Frozen OpenCLIP image encoder with tensor-native preprocessing."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        image_size: int = 224,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP conditioning requires `pip install open_clip_torch`."
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.eval()
        self.image_size = image_size
        self.register_buffer(
            "image_mean",
            torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "OpenCLIPImageEncoder":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.image_mean) / self.image_std
        embedding = self.model.encode_image(image)
        return F.normalize(embedding.float(), dim=-1)


class NullSemanticEncoder(nn.Module):
    """A no-download fallback for tests and architecture smoke runs."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, image: Tensor) -> Tensor:
        return image.new_zeros((image.shape[0], self.embedding_dim))


def build_semantic_encoder(config: dict[str, Any]) -> nn.Module:
    encoder_type = str(config.get("encoder", "open_clip")).lower()
    embedding_dim = int(config.get("embedding_dim", 512))
    if encoder_type in {"none", "null"}:
        return NullSemanticEncoder(embedding_dim)
    if encoder_type == "open_clip":
        return OpenCLIPImageEncoder(
            model_name=str(config.get("model_name", "ViT-B-32")),
            pretrained=str(config.get("pretrained", "laion2b_s34b_b79k")),
            image_size=int(config.get("image_size", 224)),
        )
    raise ValueError(f"Unsupported semantic encoder: {encoder_type}")


class DegradationAwareCrossAttention(nn.Module):
    """Cross-attend image features to learned tokens from a CLIP embedding.

    The paper specifies one global descriptor but does not report tokenization.
    Expanding it into a small token set avoids the degenerate one-key softmax
    while retaining the stated frozen global CLIP prior.
    """

    def __init__(
        self,
        channels: int,
        semantic_dim: int,
        num_heads: int = 1,
        context_tokens: int = 4,
        expansion_factor: float = 2.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(
                f"channels ({channels}) must be divisible by heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.context_tokens = context_tokens
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = LayerNorm2d(channels, bias=True)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.semantic_to_kv = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, context_tokens * channels * 2, bias=bias),
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.ffn_norm = LayerNorm2d(channels, bias=True)
        self.ffn = GatedDconvFeedForward(
            channels, expansion_factor=expansion_factor, bias=bias
        )

    def forward(self, feature: Tensor, semantic: Tensor) -> Tensor:
        batch, channels, height, width = feature.shape
        query = self.query(self.norm(feature))
        query = query.reshape(
            batch, self.num_heads, self.head_dim, height * width
        ).transpose(-2, -1)

        key, value = self.semantic_to_kv(semantic).reshape(
            batch, self.context_tokens, 2, self.num_heads, self.head_dim
        ).unbind(dim=2)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = attention @ value
        output = output.transpose(-2, -1).reshape(
            batch, channels, height, width
        )
        feature = feature + self.project_out(output)
        return feature + self.ffn(self.ffn_norm(feature))
