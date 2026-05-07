"""
Differentiable 3D renderer using PyTorch3D.

Pipeline:
  1. Load a textured .obj mesh
  2. Apply a (learnable) texture map to the mesh
  3. Render from arbitrary camera poses
  4. Return rendered 2D images + the 4x4 camera transform matrices

Gradients flow back to the texture map, enabling adversarial texture optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        AmbientLights,
        MeshRenderer,
        MeshRasterizer,
        RasterizationSettings,
        SoftPhongShader,
        TexturesUV,
    )
    from pytorch3d.io import load_obj
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("[renderer] PyTorch3D not found. Install with:")
    print("  pip install 'git+https://github.com/facebookresearch/pytorch3d.git'")
    print("  or: conda install pytorch3d -c pytorch3d")


class TexturedMeshRenderer(nn.Module):
    """
    Renders a textured 3D mesh from arbitrary camera poses.

    The texture map is a learnable parameter — gradients flow through
    the differentiable renderer back to the texture for optimization.

    Args:
        obj_path:        Path to .obj file (with UV unwrapping).
        texture_size:    Spatial resolution of the texture map (H=W).
        image_size:      Output rendered image resolution (H=W).
        device:          torch device.
        init_texture:    Optional (H, W, 3) tensor to initialize texture.
                         If None, initializes from .obj's .mtl texture if present,
                         otherwise random uniform RGB.
    """

    def __init__(
        self,
        obj_path: str,
        texture_size: int = 256,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        init_texture: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not HAS_PYTORCH3D:
            raise RuntimeError("PyTorch3D is required. See import error above.")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.texture_size = texture_size

        # Load mesh geometry (vertices, faces, UV coords)
        verts, faces, aux = load_obj(obj_path, load_textures=True, device=self.device)
        self.register_buffer("verts", verts)                        # (V, 3)
        self.register_buffer("faces_idx", faces.verts_idx)         # (F, 3)
        self.register_buffer("faces_tex_idx", faces.textures_idx)  # (F, 3)
        self.register_buffer("verts_uvs", aux.verts_uvs)           # (T, 2)

        # Learnable texture map: (1, H, W, 3), values in [0, 1]
        if init_texture is not None:
            tex = init_texture.float().to(self.device)
            if tex.dim() == 3:
                tex = tex.unsqueeze(0)  # (1, H, W, 3)
        elif aux.texture_images:
            # Load from .mtl if available
            tex_img = list(aux.texture_images.values())[0]  # (H, W, 3) or (1,H,W,3)
            if tex_img.dim() == 3:
                tex_img = tex_img.unsqueeze(0)
            import torch.nn.functional as F
            tex = F.interpolate(
                tex_img.permute(0, 3, 1, 2).float(),
                size=(texture_size, texture_size),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # (1, H, W, 3)
        else:
            tex = torch.ones(1, texture_size, texture_size, 3, device=self.device) * 0.5

        self.texture_map = nn.Parameter(tex.to(self.device))  # differentiable

        # Build rasterizer + shader (reused across calls)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.shader = SoftPhongShader(device=self.device)

    def _build_mesh(self, batch_size: int) -> "Meshes":
        """Tile the geometry B times and attach the current texture."""
        tex_clamped = self.texture_map.clamp(0.0, 1.0)  # keep texture valid
        textures = TexturesUV(
            maps=tex_clamped.expand(batch_size, -1, -1, -1),           # (B, H, W, 3)
            faces_uvs=self.faces_tex_idx.unsqueeze(0).expand(batch_size, -1, -1),
            verts_uvs=self.verts_uvs.unsqueeze(0).expand(batch_size, -1, -1),
        )
        meshes = Meshes(
            verts=self.verts.unsqueeze(0).expand(batch_size, -1, -1),
            faces=self.faces_idx.unsqueeze(0).expand(batch_size, -1, -1),
            textures=textures,
        )
        return meshes

    def forward(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        lights: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render the mesh from a batch of camera poses.

        Args:
            R:      (B, 3, 3) rotation matrices (world-to-camera)
            T:      (B, 3)    translation vectors
            lights: PyTorch3D Lights object. Defaults to ambient-only (flat shading).

        Returns:
            images:       (B, H, W, 3) rendered RGB images in [0, 1]
            transform_4x4 (B, 4, 4) full camera transform matrices [R|T; 0 1]
        """
        B = R.shape[0]
        device = self.device

        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        if lights is None:
            lights = AmbientLights(device=device)

        meshes = self._build_mesh(B)

        # Full render
        images = self.shader(
            self.rasterizer(meshes, cameras=cameras),
            cameras=cameras,
            lights=lights,
        )  # (B, H, W, 4) — last channel is alpha

        rgb = images[..., :3]  # (B, H, W, 3)

        # Build 4×4 transform matrices for bookkeeping
        transform_4x4 = _build_transform_matrix(R, T)  # (B, 4, 4)

        return rgb, transform_4x4

    def render_with_background(
        self,
        R: torch.Tensor,
        T: torch.Tensor,
        bg: torch.Tensor,
        lights: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render and composite over a background color.

        Args:
            bg: (B, 3) background RGB in [0, 1]

        Returns:
            images:       (B, H, W, 3) composited images
            transform_4x4 (B, 4, 4)
        """
        B = R.shape[0]
        device = self.device
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        if lights is None:
            lights = AmbientLights(device=device)

        meshes = self._build_mesh(B)

        fragments = self.rasterizer(meshes, cameras=cameras)
        raw_images = self.shader(fragments, cameras=cameras, lights=lights)
        # raw_images: (B, H, W, 4)

        rgb = raw_images[..., :3]
        alpha = raw_images[..., 3:4]  # (B, H, W, 1)

        # Composite: foreground * alpha + background * (1 - alpha)
        bg_expanded = bg[:, None, None, :]  # (B, 1, 1, 3)
        composited = rgb * alpha + bg_expanded * (1.0 - alpha)

        transform_4x4 = _build_transform_matrix(R, T)
        return composited, transform_4x4


def _build_transform_matrix(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Assemble a (B, 4, 4) homogeneous transform from R (B,3,3) and T (B,3).
    """
    B = R.shape[0]
    device = R.device
    mat = torch.zeros(B, 4, 4, device=device, dtype=R.dtype)
    mat[:, :3, :3] = R
    mat[:, :3, 3] = T
    mat[:, 3, 3] = 1.0
    return mat
