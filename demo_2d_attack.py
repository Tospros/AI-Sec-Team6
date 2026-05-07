"""
Demo: EOT-based 2D adversarial attack.

Takes an image, runs the EOT PGD attack to make InceptionV3 misclassify it
as a target class, even after random 2D transformations.

Usage:
    python demo_2d_attack.py --image path/to/image.jpg --target 764
    # target 764 = "rifle" (ImageNet class)
    # target 35  = "leatherback turtle"

Without --image, generates a random RGB image for a quick sanity-check.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.classifier import InceptionV3Classifier, IMAGENET_CLASSES
from pipeline_2d.attack import EOTAttack2D, AttackConfig
from pipeline_2d.transforms import Transform2DConfig


def load_image(path: str, size: int = 299) -> torch.Tensor:
    """Load an image as (H, W, 3) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0


def random_image(size: int = 299) -> torch.Tensor:
    return torch.rand(size, size, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--target", type=int, default=IMAGENET_CLASSES["rifle"],
                        help="Target ImageNet class index (default: rifle=764)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="L-inf perturbation budget")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eot_samples", type=int, default=20)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Image ──────────────────────────────────────────────
    if args.image:
        image = load_image(args.image)
        print(f"Loaded image from {args.image}")
    else:
        image = random_image()
        print("Using random test image (no --image provided)")

    # ── Classifier ────────────────────────────────────────
    classifier = InceptionV3Classifier(device=device)

    with torch.no_grad():
        orig_logits = classifier(image.unsqueeze(0).to(device))
        orig_pred = orig_logits.argmax(dim=1).item()
    print(f"Original prediction: class {orig_pred}")

    # ── Attack ────────────────────────────────────────────
    transform_cfg = Transform2DConfig(
        rotation_deg=15.0,
        scale_min=0.9,
        scale_max=1.1,
        gaussian_noise_std=0.02,
    )
    attack_cfg = AttackConfig(
        target_class=args.target,
        epsilon=args.epsilon,
        step_size=args.epsilon / 20,
        num_steps=args.steps,
        eot_samples=args.eot_samples,
        log_every=50,
    )

    attacker = EOTAttack2D(classifier, transform_cfg, attack_cfg, device=device)

    print(f"\nRunning EOT attack → target class {args.target}...")
    adv_image, history = attacker.attack(image)

    # ── Evaluate ──────────────────────────────────────────
    with torch.no_grad():
        adv_logits = classifier(adv_image.unsqueeze(0).to(device))
        adv_pred = adv_logits.argmax(dim=1).item()
    print(f"\nAdversarial prediction: class {adv_pred}")
    print(f"Attack {'SUCCEEDED' if adv_pred == args.target else 'FAILED'}")

    perturbation = (adv_image - image.to(device)).abs().max().item()
    print(f"Max L-inf perturbation: {perturbation:.4f} (budget: {args.epsilon})")

    # ── Visualise ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.numpy().clip(0, 1))
    axes[0].set_title(f"Original (pred={orig_pred})")
    axes[0].axis("off")

    axes[1].imshow(adv_image.cpu().numpy().clip(0, 1))
    axes[1].set_title(f"Adversarial (pred={adv_pred})")
    axes[1].axis("off")

    diff = ((adv_image.cpu() - image) * 10 + 0.5).clamp(0, 1)
    axes[2].imshow(diff.numpy())
    axes[2].set_title("Perturbation ×10")
    axes[2].axis("off")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
