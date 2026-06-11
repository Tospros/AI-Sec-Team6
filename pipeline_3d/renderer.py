"""
3D renderer with PyTorch3D (preferred) or pure-PyTorch/NumPy software fallback.

Software fallback:
  - Loads .obj with a built-in parser (no extra deps)
  - Z-buffer rasterizer in NumPy (not differentiable)
  - Texture sampling via F.grid_sample (differentiable — gradients reach texture_map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    print("[renderer] PyTorch3D not found — using software renderer fallback.")
    print("  To enable PyTorch3D: pip install 'git+https://github.com/facebookresearch/pytorch3d.git'")


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_transform_matrix(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    B = R.shape[0]
    mat = torch.zeros(B, 4, 4, device=R.device, dtype=R.dtype)
    mat[:, :3, :3] = R
    mat[:, :3, 3] = T
    mat[:, 3, 3] = 1.0
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch3D renderer (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class _Pytorch3DTexturedMeshRenderer(nn.Module):
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
    """

    def __init__(
        self,
        obj_path: str,
        texture_size: int = 256,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        init_texture: Optional[torch.Tensor] = None,
        normalize_mesh: bool = True,
        fov_deg: float = 60.0,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.texture_size = texture_size
        self.fov_deg = fov_deg

        verts, faces, aux = load_obj(obj_path, load_textures=True, device=self.device)

        # Center at origin and scale to unit bounding-sphere radius (matches the
        # software backend so both render the object at the same framing).
        if normalize_mesh and verts.numel():
            mn, mx = verts.min(0).values, verts.max(0).values
            center = (mn + mx) / 2.0
            radius = torch.linalg.norm((mx - mn) / 2.0)
            if radius > 0:
                verts = (verts - center) / radius

        self.register_buffer("verts", verts)
        self.register_buffer("faces_idx", faces.verts_idx)
        self.register_buffer("faces_tex_idx", faces.textures_idx)
        self.register_buffer("verts_uvs", aux.verts_uvs)

        if init_texture is not None:
            tex = init_texture.float().to(self.device)
            if tex.dim() == 3:
                tex = tex.unsqueeze(0)
        elif aux.texture_images:
            tex_img = list(aux.texture_images.values())[0]
            if tex_img.dim() == 3:
                tex_img = tex_img.unsqueeze(0)
            tex = F.interpolate(
                tex_img.permute(0, 3, 1, 2).float(),
                size=(texture_size, texture_size),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            tex = torch.ones(1, texture_size, texture_size, 3, device=self.device) * 0.5

        self.texture_map = nn.Parameter(tex.to(self.device))

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.shader = SoftPhongShader(device=self.device)

    def _build_mesh(self, batch_size: int) -> "Meshes":
        tex_clamped = self.texture_map.clamp(0.0, 1.0)
        textures = TexturesUV(
            maps=tex_clamped.expand(batch_size, -1, -1, -1),
            faces_uvs=self.faces_tex_idx.unsqueeze(0).expand(batch_size, -1, -1),
            verts_uvs=self.verts_uvs.unsqueeze(0).expand(batch_size, -1, -1),
        )
        return Meshes(
            verts=self.verts.unsqueeze(0).expand(batch_size, -1, -1),
            faces=self.faces_idx.unsqueeze(0).expand(batch_size, -1, -1),
            textures=textures,
        )

    def forward(self, R, T, lights=None):
        B = R.shape[0]
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=self.fov_deg, device=self.device)
        if lights is None:
            lights = AmbientLights(device=self.device)
        meshes = self._build_mesh(B)
        images = self.shader(
            self.rasterizer(meshes, cameras=cameras),
            cameras=cameras, lights=lights,
        )
        return images[..., :3], _build_transform_matrix(R, T)

    def render_with_background(self, R, T, bg, lights=None):
        B = R.shape[0]
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=self.fov_deg, device=self.device)
        if lights is None:
            lights = AmbientLights(device=self.device)
        meshes = self._build_mesh(B)
        fragments = self.rasterizer(meshes, cameras=cameras)
        raw = self.shader(fragments, cameras=cameras, lights=lights)
        rgb, alpha = raw[..., :3], raw[..., 3:4]
        composited = rgb * alpha + bg[:, None, None, :] * (1.0 - alpha)
        return composited, _build_transform_matrix(R, T)


# ──────────────────────────────────────────────────────────────────────────────
# Software renderer (no PyTorch3D)
# ──────────────────────────────────────────────────────────────────────────────

def _load_obj_simple(obj_path: str):
    """Minimal .obj parser. Returns numpy arrays for verts, face indices, UVs."""
    verts, uvs = [], []
    faces_v, faces_uv = [], []
    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'f':
                fv, ft = [], []
                for tok in parts[1:]:
                    idx = tok.split('/')
                    fv.append(int(idx[0]) - 1)
                    ft.append(int(idx[1]) - 1 if len(idx) > 1 and idx[1] else 0)
                for i in range(1, len(fv) - 1):
                    faces_v.append([fv[0], fv[i], fv[i + 1]])
                    faces_uv.append([ft[0], ft[i], ft[i + 1]])

    verts_np = np.array(verts, dtype=np.float32)
    faces_v_np = np.array(faces_v, dtype=np.int64)
    uvs_np = np.array(uvs, dtype=np.float32) if uvs else np.zeros((1, 2), dtype=np.float32)
    faces_uv_np = np.array(faces_uv, dtype=np.int64) if faces_uv else np.zeros_like(faces_v_np)
    return verts_np, faces_v_np, uvs_np, faces_uv_np


def _load_diffuse_texture(obj_path: str, texture_size: int) -> Optional[torch.Tensor]:
    """
    Resolve the diffuse texture (map_Kd) referenced by the .obj's mtllib and load it.

    Filenames may contain spaces (e.g. "... v1 Textured.mtl"), so the value is taken
    as the full remainder of the line after the keyword rather than a single token.

    Returns (1, texture_size, texture_size, 3) float tensor in [0, 1], or None if no
    usable texture is found.
    """
    import os
    from PIL import Image

    obj_dir = os.path.dirname(os.path.abspath(obj_path))

    def _value_after(line: str, keyword: str) -> Optional[str]:
        stripped = line.strip()
        if stripped.lower().startswith(keyword.lower()):
            return stripped[len(keyword):].strip()
        return None

    # Collect mtllib references from the .obj
    mtl_files = []
    with open(obj_path) as f:
        for line in f:
            v = _value_after(line, "mtllib")
            if v:
                mtl_files.append(v)

    # Find the first map_Kd across the referenced .mtl files
    tex_name = None
    for mtl in mtl_files:
        mtl_path = os.path.join(obj_dir, mtl)
        if not os.path.exists(mtl_path):
            continue
        with open(mtl_path) as f:
            for line in f:
                v = _value_after(line, "map_Kd")
                if v:
                    tex_name = v
                    break
        if tex_name:
            break

    if not tex_name:
        return None

    tex_path = os.path.join(obj_dir, tex_name)
    if not os.path.exists(tex_path):
        return None

    img = Image.open(tex_path).convert("RGB").resize((texture_size, texture_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
    return torch.from_numpy(arr).unsqueeze(0)          # (1, H, W, 3)


def _rasterize(verts_2d: np.ndarray, depths: np.ndarray,
               faces_v: np.ndarray, faces_uv: np.ndarray,
               verts_uvs: np.ndarray, H: int, W: int):
    """
    Z-buffer rasterizer. Returns uv_map (H,W,2) and mask (H,W) bool.
    Not differentiable — only determines which UV coords map to which pixels.
    """
    uv_map = np.zeros((H, W, 2), dtype=np.float32)
    z_buf = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    p = verts_2d

    for fi in range(len(faces_v)):
        i0, i1, i2 = faces_v[fi]
        t0, t1, t2 = faces_uv[fi]
        p0, p1, p2 = p[i0], p[i1], p[i2]

        # Bounding box
        x_lo = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x_hi = min(W - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        y_lo = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y_hi = min(H - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if x_lo > x_hi or y_lo > y_hi:
            continue

        e0x, e0y = p1[0] - p0[0], p1[1] - p0[1]
        e1x, e1y = p2[0] - p0[0], p2[1] - p0[1]
        denom = e0x * e1y - e1x * e0y
        if abs(denom) < 1e-8:
            continue
        inv_d = 1.0 / denom

        xs = np.arange(x_lo, x_hi + 1, dtype=np.float32) + 0.5
        ys = np.arange(y_lo, y_hi + 1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys)

        dx, dy = px - p0[0], py - p0[1]
        b1 = (dx * e1y - e1x * dy) * inv_d   # weight for p1
        b2 = (e0x * dy - dx * e0y) * inv_d   # weight for p2
        b0 = 1.0 - b1 - b2                   # weight for p0

        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not inside.any():
            continue

        z_interp = b0 * depths[i0] + b1 * depths[i1] + b2 * depths[i2]
        iy = py.astype(np.int32)
        ix = px.astype(np.int32)
        update = inside & (z_interp < z_buf[iy, ix])

        iy_u, ix_u = iy[update], ix[update]
        b0_u, b1_u, b2_u = b0[update], b1[update], b2[update]
        uv_map[iy_u, ix_u, 0] = b0_u * verts_uvs[t0, 0] + b1_u * verts_uvs[t1, 0] + b2_u * verts_uvs[t2, 0]
        uv_map[iy_u, ix_u, 1] = b0_u * verts_uvs[t0, 1] + b1_u * verts_uvs[t1, 1] + b2_u * verts_uvs[t2, 1]
        z_buf[iy_u, ix_u] = z_interp[update]
        mask[iy_u, ix_u] = True

    return uv_map, mask


def _project_verts(verts: np.ndarray, R: torch.Tensor, T: torch.Tensor,
                   H: int, W: int, fov_deg: float = 60.0):
    """
    Project 3D vertices to 2D pixel coordinates.
    Returns verts_2d (V,2) and depths (V,) in camera space.
    """
    v = torch.from_numpy(verts).to(R.device)           # (V, 3)
    v_cam = (R @ v.T).T + T                             # (V, 3)

    depths_t = v_cam[:, 2]
    depths = depths_t.detach().cpu().numpy()

    f = 1.0 / np.tan(np.radians(fov_deg / 2.0))
    z = depths_t.detach().cpu().numpy()
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    x_ndc = f * v_cam[:, 0].detach().cpu().numpy() / safe_z
    y_ndc = f * v_cam[:, 1].detach().cpu().numpy() / safe_z

    x_px = (x_ndc + 1.0) * 0.5 * W
    y_px = (-y_ndc + 1.0) * 0.5 * H   # flip Y: NDC up = image down

    verts_2d = np.stack([x_px, y_px], axis=1).astype(np.float32)
    return verts_2d, depths


class SoftwareTexturedMeshRenderer(nn.Module):
    """
    Pure PyTorch/NumPy textured mesh renderer — no PyTorch3D required.

    Rasterization is CPU NumPy (no grad); texture sampling uses F.grid_sample
    so gradients DO flow back to texture_map for adversarial optimization.
    """

    def __init__(
        self,
        obj_path: str,
        texture_size: int = 256,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        init_texture: Optional[torch.Tensor] = None,
        normalize_mesh: bool = True,
        fov_deg: float = 60.0,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.image_size = image_size
        self.texture_size = texture_size
        self.fov_deg = fov_deg

        verts_np, faces_v_np, uvs_np, faces_uv_np = _load_obj_simple(obj_path)

        # Center at origin and scale to unit bounding-sphere radius so the object
        # is consistently framed regardless of the .obj's modelling units/offset.
        if normalize_mesh and len(verts_np):
            mn, mx = verts_np.min(0), verts_np.max(0)
            center = (mn + mx) / 2.0
            radius = float(np.linalg.norm((mx - mn) / 2.0))
            if radius > 0:
                verts_np = ((verts_np - center) / radius).astype(np.float32)

        self.verts_np = verts_np
        self.faces_v_np = faces_v_np
        self.uvs_np = uvs_np
        self.faces_uv_np = faces_uv_np

        if init_texture is not None:
            tex = init_texture.float()
            if tex.dim() == 3:
                tex = tex.unsqueeze(0)
        else:
            tex = _load_diffuse_texture(obj_path, texture_size)
            if tex is None:
                tex = torch.ones(1, texture_size, texture_size, 3) * 0.5

        self.texture_map = nn.Parameter(tex.to(self.device))

    def _render_single(self, R: torch.Tensor, T: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        H = W = self.image_size
        verts_2d, depths = _project_verts(self.verts_np, R, T, H, W, fov_deg=self.fov_deg)
        uv_map, mask = _rasterize(
            verts_2d, depths,
            self.faces_v_np, self.faces_uv_np,
            self.uvs_np, H, W,
        )
        return self._sample_texture(uv_map, mask, bg, H, W)

    def _sample_texture(self, uv_map_np, mask_np, bg, H, W):
        uv_t = torch.from_numpy(uv_map_np).to(self.device)   # (H, W, 2)
        # Convert UV [0,1] -> grid_sample [-1,1]; flip V (obj UV origin = bottom-left)
        grid_x = uv_t[..., 0] * 2.0 - 1.0
        grid_y = 1.0 - uv_t[..., 1] * 2.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        tex = self.texture_map.clamp(0.0, 1.0).permute(0, 3, 1, 2)  # (1, 3, Ht, Wt)
        sampled = F.grid_sample(tex, grid, mode='bilinear', align_corners=False, padding_mode='border')
        sampled = sampled.squeeze(0).permute(1, 2, 0)   # (H, W, 3)

        mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(-1)   # (H, W, 1)
        bg_exp = bg.view(1, 1, 3).expand(H, W, 3)
        return torch.where(mask_t, sampled, bg_exp)   # (H, W, 3)

    def forward(self, R, T, lights=None):
        B = R.shape[0]
        bg = torch.zeros(B, 3, device=self.device)
        images = torch.stack([self._render_single(R[i], T[i], bg[i]) for i in range(B)])
        return images, _build_transform_matrix(R, T)

    def render_with_background(self, R, T, bg, lights=None):
        B = R.shape[0]
        images = torch.stack([self._render_single(R[i], T[i], bg[i]) for i in range(B)])
        return images, _build_transform_matrix(R, T)


# ──────────────────────────────────────────────────────────────────────────────
# Public factory — picks the right backend automatically
# ──────────────────────────────────────────────────────────────────────────────

def TexturedMeshRenderer(
    obj_path: str,
    texture_size: int = 256,
    image_size: int = 224,
    device: Optional[torch.device] = None,
    init_texture: Optional[torch.Tensor] = None,
    normalize_mesh: bool = True,
    fov_deg: float = 60.0,
) -> nn.Module:
    """
    Returns a PyTorch3D renderer when available, otherwise the software fallback.
    Both implement the same interface: forward(R, T) and render_with_background(R, T, bg).

    fov_deg: camera field of view. Lower = telephoto (object appears larger with
             flatter perspective) — useful to make a small object fill the frame.
    """
    cls = _Pytorch3DTexturedMeshRenderer if HAS_PYTORCH3D else SoftwareTexturedMeshRenderer
    return cls(obj_path, texture_size=texture_size, image_size=image_size,
               device=device, init_texture=init_texture,
               normalize_mesh=normalize_mesh, fov_deg=fov_deg)
