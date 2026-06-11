"""
Random 3D transformation sampling for the EOT (Expectation Over Transformation) framework.

Based on Table 5 from "Synthesizing Robust Adversarial Examples" (Athalye et al., 2017).
Produces camera poses as 4x4 transformation matrices.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TransformConfig:
    """
    Transformation distribution parameters.
    Defaults match Table 5 (simulation domain) of the paper.
    """
    # Camera distance from object origin
    dist_min: float = 2.5
    dist_max: float = 3.0

    # XY translation of object (in world units)
    translate_xy_range: float = 0.05

    # Rotation sampling mode:
    #   "so3"        — uniform over all SO(3)  (default)
    #   "yaw"        — yaw-only, within ±rotation_y_range_deg
    #   "yaw_pitch"  — yaw within [yaw_min_deg, yaw_max_deg] and
    #                  pitch within [pitch_min_deg, pitch_max_deg]
    rotation_mode: str = "so3"
    rotation_y_range_deg: float = 180.0       # used by "yaw"
    yaw_min_deg: float = -180.0               # used by "yaw_pitch"
    yaw_max_deg: float = 180.0
    pitch_min_deg: float = 0.0
    pitch_max_deg: float = 0.0

    # Legacy flag: when False (and rotation_mode left at default) → yaw-only.
    full_rotation: bool = True

    # Background color (uniform random RGB per channel)
    bg_color_min: float = 0.1
    bg_color_max: float = 1.0

    # Physical-world augmentation (Table 6) — disabled by default
    additive_light: float = 0.0       # ±value
    mult_light_min: float = 1.0       # per-channel
    mult_light_max: float = 1.0
    color_noise_add: float = 0.0      # ±value
    color_noise_mult_min: float = 1.0
    color_noise_mult_max: float = 1.0
    gaussian_noise_std: float = 0.0   # image-space Gaussian noise


def sample_rotation_matrix(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample uniformly random rotation matrices from SO(3).
    Uses the QR decomposition method for uniform sampling.

    Returns: (B, 3, 3) rotation matrices
    """
    # Sample random normal matrices and QR-decompose for uniform SO(3) coverage
    z = torch.randn(batch_size, 3, 3, device=device)
    q, r = torch.linalg.qr(z)
    # Ensure det = +1 (not -1)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))  # (B, 3)
    q = q * sign.unsqueeze(1)  # flip columns where diagonal of R < 0
    return q  # (B, 3, 3)


def sample_yaw_rotation(batch_size: int, range_deg: float, device: torch.device) -> torch.Tensor:
    """
    Sample yaw-only rotation matrices (rotation around Y axis).

    Returns: (B, 3, 3) rotation matrices
    """
    angles = torch.empty(batch_size, device=device).uniform_(-range_deg, range_deg)
    angles_rad = angles * (torch.pi / 180.0)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    # Yaw rotation matrix (around Y)
    R = torch.stack([
        torch.stack([cos_a, zeros, sin_a], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin_a, zeros, cos_a], dim=1),
    ], dim=1)  # (B, 3, 3)
    return R


def sample_yaw_pitch_rotation(
    batch_size: int,
    yaw_min: float, yaw_max: float,
    pitch_min: float, pitch_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample rotations as R = Rx(pitch) @ Ry(yaw), with yaw/pitch drawn uniformly
    from the given (degree) ranges. Yaw spins the object; pitch tilts the camera
    up/down. Returns (B, 3, 3).
    """
    yaw = torch.empty(batch_size, device=device).uniform_(yaw_min, yaw_max) * (torch.pi / 180.0)
    pitch = torch.empty(batch_size, device=device).uniform_(pitch_min, pitch_max) * (torch.pi / 180.0)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    z, o = torch.zeros_like(cy), torch.ones_like(cy)

    Ry = torch.stack([
        torch.stack([cy, z, sy], dim=1),
        torch.stack([z, o, z], dim=1),
        torch.stack([-sy, z, cy], dim=1),
    ], dim=1)
    Rx = torch.stack([
        torch.stack([o, z, z], dim=1),
        torch.stack([z, cp, -sp], dim=1),
        torch.stack([z, sp, cp], dim=1),
    ], dim=1)
    return Rx @ Ry


def sample_transforms(
    batch_size: int,
    cfg: TransformConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch of random camera-to-world transforms.

    Returns:
        R: (B, 3, 3)  rotation matrices (world-to-camera)
        T: (B, 3)     translation vectors (camera position in world)
        bg: (B, 3)    background RGB colors in [0, 1]
    """
    # Resolve rotation mode (honour the legacy full_rotation flag)
    mode = cfg.rotation_mode
    if mode == "so3" and not cfg.full_rotation:
        mode = "yaw"

    if mode == "yaw_pitch":
        R = sample_yaw_pitch_rotation(
            batch_size, cfg.yaw_min_deg, cfg.yaw_max_deg,
            cfg.pitch_min_deg, cfg.pitch_max_deg, device,
        )
    elif mode == "yaw":
        R = sample_yaw_rotation(batch_size, cfg.rotation_y_range_deg, device)
    else:
        R = sample_rotation_matrix(batch_size, device)

    # Camera distance (radial)
    dist = torch.empty(batch_size, device=device).uniform_(cfg.dist_min, cfg.dist_max)

    # XY jitter around optical axis
    tx = torch.empty(batch_size, device=device).uniform_(-cfg.translate_xy_range, cfg.translate_xy_range)
    ty = torch.empty(batch_size, device=device).uniform_(-cfg.translate_xy_range, cfg.translate_xy_range)

    T = torch.stack([tx, ty, dist], dim=1)  # (B, 3)

    # Background color
    bg = torch.empty(batch_size, 3, device=device).uniform_(cfg.bg_color_min, cfg.bg_color_max)

    return R, T, bg


def apply_image_augmentation(
    images: torch.Tensor,
    cfg: TransformConfig,
) -> torch.Tensor:
    """
    Apply physical-world image-space augmentations (Table 6).
    images: (B, H, W, 3) in [0, 1]
    Returns augmented images clamped to [0, 1].
    """
    B = images.shape[0]
    device = images.device
    out = images.clone()

    if cfg.gaussian_noise_std > 0:
        out = out + torch.randn_like(out) * cfg.gaussian_noise_std

    if cfg.additive_light > 0:
        out = out + torch.empty(B, 1, 1, 3, device=device).uniform_(
            -cfg.additive_light, cfg.additive_light
        )

    if cfg.mult_light_min != 1.0 or cfg.mult_light_max != 1.0:
        scale = torch.empty(B, 1, 1, 3, device=device).uniform_(
            cfg.mult_light_min, cfg.mult_light_max
        )
        out = out * scale

    if cfg.color_noise_add > 0:
        out = out + torch.empty(B, 1, 1, 3, device=device).uniform_(
            -cfg.color_noise_add, cfg.color_noise_add
        )

    if cfg.color_noise_mult_min != 1.0 or cfg.color_noise_mult_max != 1.0:
        scale = torch.empty(B, 1, 1, 3, device=device).uniform_(
            cfg.color_noise_mult_min, cfg.color_noise_mult_max
        )
        out = out * scale

    return out.clamp(0.0, 1.0)
