"""
Demo: 3D adversarial texture attack (EOT).

Optimizes the texture of a 3D mesh so that InceptionV3 misclassifies
the rendered object from arbitrary viewpoints.

Usage:
    # Quick test with demo cube, untargeted attack:
    python demo_3d_attack.py --demo_mesh

    # Targeted attack (make cube look like a rifle, class 764):
    python demo_3d_attack.py --demo_mesh --target 764

    # With a real mesh:
    python demo_3d_attack.py --obj turtle.obj --target 764 --steps 500 --eot 40
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline_3d.renderer import TexturedMeshRenderer, SoftwareTexturedMeshRenderer
from pipeline_3d.transforms import TransformConfig, sample_transforms
from pipeline_3d.attack import EOTMaskAttack3D, Attack3DConfig
from models.classifier import InceptionV3Classifier, IMAGENET_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# Demo mesh helper (copy from demo_3d_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def create_demo_obj(path: str):
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def render_views(renderer, transform_cfg, n, device):
    with torch.no_grad():
        R, T, bg = sample_transforms(n, transform_cfg, device)
        images, _ = renderer.render_with_background(R, T, bg)
    return images  # (n, H, W, 3)


def apply_mask(images, delta):
    """Overlay the screen-space mask on rendered projections: 'look through the mask'."""
    return (images + delta.to(images.device)).clamp(0.0, 1.0)


def show_views(images, classifier, title, save_path=None):
    n = images.shape[0]
    with torch.no_grad():
        logits = classifier(images)
        preds = logits.argmax(dim=1).tolist()
        probs = torch.softmax(logits, dim=1)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)
    for i, ax in enumerate(axes):
        img = images[i].cpu().numpy().clip(0, 1)
        ax.imshow(img)
        ax.axis("off")
        top_prob = probs[i, preds[i]].item()
        ax.set_title(f"cls {preds[i]}\n{top_prob:.1%}", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--demo_mesh", action="store_true")
    parser.add_argument("--target", type=int, default=None,
                        help="Target ImageNet class index (None = untargeted). "
                             "Hint: rifle=764, baseball=429, turtle=35")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eot", type=int, default=10,
                        help="EOT samples per step (more = slower but better gradients)")
    parser.add_argument("--step_size", type=float, default=0.005)
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="L-inf bound on the screen-space mask")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--texture_size", type=int, default=128)
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default="attack_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Mesh ──────────────────────────────────────────────────────────────────
    if args.obj:
        obj_path = args.obj
    elif args.demo_mesh:
        import tempfile
        obj_path = create_demo_obj(str(Path(tempfile.gettempdir()) / "demo_cube.obj"))
        print(f"Demo cube: {obj_path}")
    else:
        print("Provide --obj <path> or --demo_mesh")
        return

    # ── Renderer + Classifier ─────────────────────────────────────────────────
    renderer = TexturedMeshRenderer(
        obj_path=obj_path,
        texture_size=args.texture_size,
        image_size=args.image_size,
        device=device,
    )
    print(f"Texture shape: {renderer.texture_map.shape}")

    print("Loading InceptionV3...")
    classifier = InceptionV3Classifier(device=device)

    transform_cfg = TransformConfig()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ── Before attack ─────────────────────────────────────────────────────────
    print(f"\n--- Before attack ---")
    images_before = render_views(renderer, transform_cfg, args.n_views, device)
    show_views(images_before, classifier, "Before attack",
               save_path=str(save_dir / "before.png"))

    # ── Attack ────────────────────────────────────────────────────────────────
    attack_cfg = Attack3DConfig(
        target_class=args.target,
        epsilon=args.epsilon,
        step_size=args.step_size,
        num_steps=args.steps,
        eot_samples=args.eot,
        log_every=max(1, args.steps // 10),
    )

    mode = f"targeted -> class {args.target}" if args.target is not None else "untargeted"
    print(f"\n--- Attack: {mode} | steps={args.steps} eot={args.eot} ---")

    attack = EOTMaskAttack3D(
        classifier=classifier,
        renderer=renderer,
        transform_cfg=transform_cfg,
        attack_cfg=attack_cfg,
        device=device,
    )
    delta, history = attack.attack()

    # ── After attack: render clean projections, then look through the mask ──────
    print(f"\n--- After attack ---")
    images_clean = render_views(renderer, transform_cfg, args.n_views, device)
    images_after = apply_mask(images_clean, delta)
    show_views(images_after, classifier, f"After attack ({mode})",
               save_path=str(save_dir / "after.png"))

    # Final fooling rate on a fresh batch of viewpoints
    fool_rate, _, _ = attack._eval_fool(max(args.n_views, 8), delta)
    metric = "target success" if args.target is not None else "fool rate"
    print(f"Final {metric}: {fool_rate:.0%}")

    # ── Save the optimized mask ────────────────────────────────────────────────
    # Mask values live in [-eps, eps]; rescale to [0, 1] for display.
    eps = max(args.epsilon, 1e-8)
    mask_vis = ((delta.squeeze(0).cpu().numpy() / (2 * eps)) + 0.5).clip(0, 1)
    plt.imsave(str(save_dir / "adv_mask.png"), mask_vis)
    print(f"Adversarial mask saved: {save_dir}/adv_mask.png")

    # ── Loss / fooling curves ──────────────────────────────────────────────────
    if len(history["loss"]) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(history["loss"])
        ax1.set_xlabel("Log step")
        ax1.set_ylabel("Loss")
        ax1.set_title("EOT 3D mask loss")
        ax2.plot(history["fool_rate"], color="tab:red")
        ax2.set_xlabel("Log step")
        ax2.set_ylabel("Rate")
        ax2.set_ylim(0, 1)
        ax2.set_title("Target success" if args.target is not None else "Fool rate")
        plt.tight_layout()
        plt.savefig(str(save_dir / "loss_curve.png"), dpi=120)
        plt.close()
        print(f"Curves saved: {save_dir}/loss_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
