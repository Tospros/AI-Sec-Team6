"""
2D image transformation distribution for the EOT 2D attack.

These transforms are applied to 2D images before classification,
making the adversarial perturbation robust to real-world conditions.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Transform2DConfig:
    """Distribution of 2D transformations to be robust to."""
    # Spatial
    scale_min: float = 0.8
    scale_max: float = 1.2
    rotation_deg: float = 30.0       # ±degrees
    translate_frac: float = 0.1      # ±fraction of image size

    # Color
    brightness: float = 0.2          # ±additive
    contrast_min: float = 0.8        # multiplicative
    contrast_max: float = 1.2
    saturation_min: float = 0.8
    saturation_max: float = 1.2

    # Noise
    gaussian_noise_std: float = 0.03

    # JPEG compression simulation (approximate via blur)
    jpeg_sim: bool = False
    jpeg_blur_std: float = 0.5


def sample_affine_matrix(
    batch_size: int,
    cfg: Transform2DConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample a batch of 2D affine matrices (for grid_sample).

    Returns: (B, 2, 3) affine matrices mapping output → input pixels.
    """
    B = batch_size

    # Scale
    scale = torch.empty(B, device=device).uniform_(cfg.scale_min, cfg.scale_max)

    # Rotation
    angle_rad = torch.empty(B, device=device).uniform_(
        -cfg.rotation_deg, cfg.rotation_deg
    ) * (torch.pi / 180.0)
    cos_a = torch.cos(angle_rad) / scale
    sin_a = torch.sin(angle_rad) / scale

    # Translation
    tx = torch.empty(B, device=device).uniform_(-cfg.translate_frac, cfg.translate_frac)
    ty = torch.empty(B, device=device).uniform_(-cfg.translate_frac, cfg.translate_frac)

    # Assemble (B, 2, 3)
    row0 = torch.stack([cos_a, -sin_a, tx], dim=1)   # (B, 3)
    row1 = torch.stack([sin_a,  cos_a, ty], dim=1)   # (B, 3)
    theta = torch.stack([row0, row1], dim=1)           # (B, 2, 3)
    return theta


def apply_spatial_transform(
    images: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Apply affine transform to images using differentiable grid_sample.

    images: (B, C, H, W)
    theta:  (B, 2, 3)
    Returns: (B, C, H, W)
    """
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="border")


def apply_color_jitter(
    images: torch.Tensor,
    cfg: Transform2DConfig,
) -> torch.Tensor:
    """
    images: (B, C, H, W) in [0, 1]
    """
    B = images.shape[0]
    device = images.device

    # Brightness
    if cfg.brightness > 0:
        images = images + torch.empty(B, 1, 1, 1, device=device).uniform_(
            -cfg.brightness, cfg.brightness
        )

    # Contrast (scale around mean)
    if cfg.contrast_min != 1.0 or cfg.contrast_max != 1.0:
        scale = torch.empty(B, 1, 1, 1, device=device).uniform_(
            cfg.contrast_min, cfg.contrast_max
        )
        mean = images.mean(dim=[2, 3], keepdim=True)
        images = (images - mean) * scale + mean

    return images.clamp(0.0, 1.0)


def apply_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std > 0:
        images = images + torch.randn_like(images) * std
    return images.clamp(0.0, 1.0)


def apply_transforms_2d(
    images: torch.Tensor,
    cfg: Transform2DConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample and apply a full set of 2D transforms in one call.

    images: (B, C, H, W) in [0, 1]

    Returns:
        transformed: (B, C, H, W)
        theta:       (B, 2, 3) — the sampled affine matrix (for logging)
    """
    B = images.shape[0]
    device = images.device

    theta = sample_affine_matrix(B, cfg, device)
    out = apply_spatial_transform(images, theta)
    out = apply_color_jitter(out, cfg)
    out = apply_noise(out, cfg.gaussian_noise_std)

    return out, theta
