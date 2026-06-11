"""
3D adversarial-mask pipeline on the sea-turtle mesh.

Pipeline:
  1. Render a few HELD-OUT projections of the turtle. The classifier sees a
     turtle (loggerhead / leatherback / ...).
  2. Optimize a single screen-space mask over the WHOLE mesh's viewpoint
     distribution (fresh random views every step — never the held-out ones) so
     that it suppresses every turtle class.
  3. Look at the held-out projections THROUGH the mask. The classifier no longer
     says turtle.

Every stage is written out as a visualization PNG to --save_dir.

Usage:
    python demo_3d_turtle_attack.py
    python demo_3d_turtle_attack.py --steps 40 --eot 8 --epsilon 0.08
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from pipeline_3d.renderer import TexturedMeshRenderer
from pipeline_3d.transforms import TransformConfig, sample_transforms
from pipeline_3d.attack import EOTMaskAttack3D, Attack3DConfig
from models.classifier import InceptionV3Classifier

# loggerhead, leatherback turtle, mud turtle, terrapin, box turtle
TURTLE_CLASSES = [33, 34, 35, 36, 37]


def turtle_view_config() -> TransformConfig:
    """Viewpoint distribution under which the clean turtle reads as a turtle.

    Paired with a telephoto FOV (see --fov), this frames the turtle large in
    the image (~20% pixel fill) so a screen-space mask must actually overpower a
    prominent, confidently-classified turtle rather than a tiny one.
    """
    return TransformConfig(
        rotation_mode="yaw_pitch",
        yaw_min_deg=170.0, yaw_max_deg=220.0,
        pitch_min_deg=55.0, pitch_max_deg=90.0,
        dist_min=2.6, dist_max=3.0,
        bg_color_min=0.5, bg_color_max=0.9,
    )


def classify(classifier, imgs, names):
    """Return (labels, conf, turtle_prob, is_turtle) for a batch of images."""
    with torch.no_grad():
        probs = torch.softmax(classifier(imgs), dim=1)
    top = probs.topk(1, dim=1)
    idx = top.indices[:, 0].tolist()
    conf = top.values[:, 0].tolist()
    tprob = probs[:, TURTLE_CLASSES].sum(dim=1).tolist()
    labels = [names[i] for i in idx]
    is_turtle = [i in TURTLE_CLASSES for i in idx]
    return labels, conf, tprob, is_turtle


def mask_to_rgb(delta, eps):
    """Rescale a mask in [-eps, eps] to a viewable [0, 1] RGB image."""
    return ((delta.squeeze(0).cpu().numpy() / (2 * max(eps, 1e-8))) + 0.5).clip(0, 1)


def save_grid(imgs, titles, is_turtle, suptitle, path, want_turtle):
    """Row of images; title green when the turtle/no-turtle goal is met, else red."""
    n = imgs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.1))
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=13)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].cpu().numpy().clip(0, 1))
        ax.axis("off")
        ok = is_turtle[i] if want_turtle else (not is_turtle[i])
        ax.set_title(titles[i], fontsize=9, color=("green" if ok else "red"))
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="turtle/20446_Sea_Turtle_v1 Textured.obj")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--eot", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--texture_size", type=int, default=256)
    parser.add_argument("--fov", type=float, default=30.0,
                        help="Camera field of view (deg). Lower = telephoto = larger turtle.")
    parser.add_argument("--save_dir", type=str, default="turtle_attack_output")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    renderer = TexturedMeshRenderer(
        obj_path=args.obj, texture_size=args.texture_size,
        image_size=args.image_size, device=device, fov_deg=args.fov,
    )
    classifier = InceptionV3Classifier(device=device)
    names = models.Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]
    cfg = turtle_view_config()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # ── Stage 1: held-out clean projections ────────────────────────────────────
    print("\n--- Stage 1: clean projections (held out) ---")
    torch.manual_seed(args.seed)
    with torch.no_grad():
        R, T, bg = sample_transforms(args.n_views, cfg, device)
        images_clean, _ = renderer.render_with_background(R, T, bg)

    labels, conf, tprob, is_turtle = classify(classifier, images_clean, names)
    for i in range(args.n_views):
        print(f"  view {i}: {labels[i]:<20} {conf[i]:.0%}  turtle_p={tprob[i]:.0%}")
    print(f"  -> {sum(is_turtle)}/{args.n_views} classified as a turtle")
    save_grid(
        images_clean,
        [f"{labels[i]}\n{conf[i]:.0%} | turtle {tprob[i]:.0%}" for i in range(args.n_views)],
        is_turtle, "Stage 1 — clean projections (classified as TURTLE)",
        save_dir / "01_clean_views.png", want_turtle=True,
    )

    # ── Stage 2: optimize the mask over the whole-mesh distribution ────────────
    print("\n--- Stage 2: optimize de-turtling mask (whole-mesh EOT) ---")
    attack_cfg = Attack3DConfig(
        protected_classes=TURTLE_CLASSES,
        epsilon=args.epsilon,
        step_size=args.step_size,
        num_steps=args.steps,
        eot_samples=args.eot,
        log_every=max(1, args.steps // 4),
    )
    torch.manual_seed(args.seed + 1)  # training views != held-out views
    attack = EOTMaskAttack3D(
        classifier=classifier, renderer=renderer,
        transform_cfg=cfg, attack_cfg=attack_cfg, device=device,
    )
    delta, history = attack.attack()

    # ── Stage 3: look at the held-out projections through the mask ─────────────
    print("\n--- Stage 3: held-out projections through the mask ---")
    images_masked = (images_clean + delta).clamp(0.0, 1.0)
    m_labels, m_conf, m_tprob, m_is_turtle = classify(classifier, images_masked, names)
    for i in range(args.n_views):
        print(f"  view {i}: {m_labels[i]:<20} {m_conf[i]:.0%}  turtle_p={m_tprob[i]:.0%}")
    de_turtled = sum(not t for t in m_is_turtle)
    print(f"  -> {de_turtled}/{args.n_views} no longer classified as a turtle")

    # ── Visualizations ─────────────────────────────────────────────────────────
    print("\n--- Saving visualizations ---")
    # 02: the universal mask
    plt.figure(figsize=(3.4, 3.4))
    plt.imshow(mask_to_rgb(delta, args.epsilon))
    plt.axis("off")
    plt.title(f"Stage 2 — optimized mask\n(L-inf eps={args.epsilon}, amplified)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "02_mask.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_dir/'02_mask.png'}")

    # 03: masked held-out views
    save_grid(
        images_masked,
        [f"{m_labels[i]}\n{m_conf[i]:.0%} | turtle {m_tprob[i]:.0%}" for i in range(args.n_views)],
        m_is_turtle, "Stage 3 — through the mask (no longer TURTLE)",
        save_dir / "03_masked_views.png", want_turtle=False,
    )

    # 04: before/after comparison (2 rows)
    fig, axes = plt.subplots(2, args.n_views, figsize=(2.7 * args.n_views, 6.0))
    for i in range(args.n_views):
        axes[0, i].imshow(images_clean[i].cpu().numpy().clip(0, 1)); axes[0, i].axis("off")
        axes[0, i].set_title(f"{labels[i]}\nturtle {tprob[i]:.0%}", fontsize=9,
                             color="green" if is_turtle[i] else "red")
        axes[1, i].imshow(images_masked[i].cpu().numpy().clip(0, 1)); axes[1, i].axis("off")
        axes[1, i].set_title(f"{m_labels[i]}\nturtle {m_tprob[i]:.0%}", fontsize=9,
                             color="red" if m_is_turtle[i] else "green")
    axes[0, 0].set_ylabel("clean", fontsize=12)
    axes[1, 0].set_ylabel("masked", fontsize=12)
    fig.suptitle("Clean (top) vs. through-the-mask (bottom)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "04_before_after.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_dir/'04_before_after.png'}")

    # 05: triptych for one view — clean + mask = masked
    v = 0
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(images_clean[v].cpu().numpy().clip(0, 1)); axes[0].axis("off")
    axes[0].set_title(f"projection\n{labels[v]} (turtle {tprob[v]:.0%})", fontsize=10, color="green")
    axes[1].imshow(mask_to_rgb(delta, args.epsilon)); axes[1].axis("off")
    axes[1].set_title("+ adversarial mask", fontsize=10)
    axes[2].imshow(images_masked[v].cpu().numpy().clip(0, 1)); axes[2].axis("off")
    axes[2].set_title(f"= fooled\n{m_labels[v]} (turtle {m_tprob[v]:.0%})", fontsize=10,
                      color="red" if m_is_turtle[v] else "green")
    fig.suptitle("What happens in between: projection + mask -> fooled", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "05_triptych.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_dir/'05_triptych.png'}")

    # 06: optimization curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.plot(history["step"], history["protected_prob"], marker="o", color="tab:green")
    ax1.set_xlabel("step"); ax1.set_ylabel("mean turtle probability")
    ax1.set_ylim(0, 1); ax1.set_title("Turtle probability ↓ during optimization")
    ax2.plot(history["step"], history["fool_rate"], marker="o", color="tab:red")
    ax2.set_xlabel("step"); ax2.set_ylabel("de-turtle rate")
    ax2.set_ylim(0, 1); ax2.set_title("Fraction no longer turtle ↑")
    plt.tight_layout()
    plt.savefig(save_dir / "06_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_dir/'06_curves.png'}")

    print(f"\nDone. Clean: {sum(is_turtle)}/{args.n_views} turtle  ->  "
          f"masked: {de_turtled}/{args.n_views} de-turtled. Files in {save_dir}/")


if __name__ == "__main__":
    main()
