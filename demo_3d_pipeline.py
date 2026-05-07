"""
Demo: 3D pipeline — load a textured mesh, render it from random viewpoints,
      and display the resulting 2D projections with their transform matrices.

Usage:
    python demo_3d_pipeline.py --obj path/to/mesh.obj --n_views 8

For a quick test without a real mesh, run with --demo_mesh to generate
a simple UV-textured cube on-the-fly.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline_3d.transforms import TransformConfig, sample_transforms
from pipeline_3d.renderer import TexturedMeshRenderer


# ──────────────────────────────────────────────────────────
# Demo mesh generation (no .obj file needed)
# ──────────────────────────────────────────────────────────

def create_demo_obj(path: str = "/tmp/demo_cube.obj"):
    """Write a minimal UV-mapped cube .obj to disk for testing."""
    obj_content = """\
# Minimal UV-mapped cube
mtllib demo_cube.mtl

v  1  1 -1
v  1 -1 -1
v  1  1  1
v  1 -1  1
v -1  1 -1
v -1 -1 -1
v -1  1  1
v -1 -1  1

vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1

vn  0  1  0
vn  0  0  1
vn -1  0  0
vn  0 -1  0
vn  1  0  0
vn  0  0 -1

usemtl demo
f 1/1/1 5/2/1 7/3/1 3/4/1
f 4/1/2 3/2/2 7/3/2 8/4/2
f 8/1/3 7/2/3 5/3/3 6/4/3
f 6/5/4 2/6/4 4/7/4 8/8/4
f 2/1/5 1/2/5 3/3/5 4/4/5
f 6/5/6 5/6/6 1/7/6 2/8/6
"""
    mtl_content = """\
newmtl demo
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
d 1
Ns 0
illum 1
"""
    p = Path(path)
    p.write_text(obj_content)
    (p.parent / "demo_cube.mtl").write_text(mtl_content)
    return str(p)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None,
                        help="Path to .obj file with UV mapping")
    parser.add_argument("--demo_mesh", action="store_true",
                        help="Generate a simple demo cube mesh")
    parser.add_argument("--n_views", type=int, default=6,
                        help="Number of random viewpoints to render")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--texture_size", type=int, default=256)
    parser.add_argument("--save", type=str, default=None,
                        help="Save rendered images to this path (e.g. renders.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Mesh ──────────────────────────────────────────────
    if args.obj:
        obj_path = args.obj
    elif args.demo_mesh:
        import tempfile
        obj_path = create_demo_obj(str(Path(tempfile.gettempdir()) / "demo_cube.obj"))
        print(f"Created demo cube at: {obj_path}")
    else:
        print("No .obj file provided. Use --obj <path> or --demo_mesh.")
        print("Example: python demo_3d_pipeline.py --demo_mesh --n_views 6")
        return

    # ── Renderer ──────────────────────────────────────────
    renderer = TexturedMeshRenderer(
        obj_path=obj_path,
        texture_size=args.texture_size,
        image_size=args.image_size,
        device=device,
    )
    print(f"Loaded mesh. Texture shape: {renderer.texture_map.shape}")

    # ── Transform config (paper's Table 5 defaults) ────────
    transform_cfg = TransformConfig()

    # ── Sample random viewpoints ──────────────────────────
    B = args.n_views
    R, T, bg = sample_transforms(B, transform_cfg, device)

    print(f"\nRendering {B} views...")
    with torch.no_grad():
        images, transforms = renderer.render_with_background(R, T, bg)
    # images: (B, H, W, 3)   transforms: (B, 4, 4)

    print(f"Rendered images shape: {images.shape}")
    print(f"Transform matrices shape: {transforms.shape}")

    # Print first transform matrix as example
    print(f"\nExample transform matrix (view 0):\n{transforms[0].cpu().numpy()}")

    # ── Visualise ─────────────────────────────────────────
    fig, axes = plt.subplots(1, B, figsize=(3 * B, 3))
    if B == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = images[i].cpu().numpy().clip(0, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"View {i}")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
