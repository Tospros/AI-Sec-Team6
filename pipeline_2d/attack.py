"""
2D EOT (Expectation Over Transformation) adversarial attack.

Optimizes an adversarial perturbation that remains effective after
random 2D image transformations (rotation, scale, color jitter, noise).

Reference: Athalye et al. 2017 - "Synthesizing Robust Adversarial Examples"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from dataclasses import dataclass, field

from pipeline_2d.transforms import Transform2DConfig, apply_transforms_2d


@dataclass
class AttackConfig:
    # Target class index (None = untargeted attack)
    target_class: Optional[int] = None

    # Perturbation constraint
    epsilon: float = 0.05          # max L-inf norm of perturbation
    step_size: float = 0.005       # PGD step size
    num_steps: int = 500           # PGD iterations

    # EOT
    eot_samples: int = 40          # number of transforms per gradient step

    # Logging
    log_every: int = 50


class EOTAttack2D:
    """
    Projected Gradient Descent adversarial attack with EOT for 2D images.

    Optimizes:
        delta = argmax  E_{t ~ T}[ loss(f(t(x + delta)), y_target) ]
        s.t.   ||delta||_inf <= epsilon

    For a targeted attack, `loss` is -cross_entropy (maximize target class).
    For untargeted, `loss` is +cross_entropy (maximize prediction error).

    Args:
        classifier: callable (B, H, W, 3) → (B, num_classes) logits.
                    Images are expected in [0, 1] BHWC format.
        transform_cfg: distribution of 2D augmentations.
        attack_cfg: attack hyperparameters.
        device: torch device.
    """

    def __init__(
        self,
        classifier: Callable,
        transform_cfg: Optional[Transform2DConfig] = None,
        attack_cfg: Optional[AttackConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.classifier = classifier
        self.transform_cfg = transform_cfg or Transform2DConfig()
        self.cfg = attack_cfg or AttackConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run the EOT attack on a single image.

        Args:
            image: (H, W, 3) or (1, H, W, 3) tensor in [0, 1]

        Returns:
            adv_image: (H, W, 3) adversarial image in [0, 1]
            info:      dict with loss/pred history
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, H, W, 3)
        image = image.float().to(self.device)

        # Convert to CHW for spatial transforms
        x = image.permute(0, 3, 1, 2)  # (1, C, H, W)

        # Initialize perturbation uniformly in [-eps, eps]
        delta = torch.empty_like(x).uniform_(-self.cfg.epsilon, self.cfg.epsilon)
        delta.requires_grad_(True)

        history = {"loss": [], "pred": []}

        for step in range(self.cfg.num_steps):
            # Accumulate gradients over EOT samples
            total_loss = torch.tensor(0.0, device=self.device)

            for _ in range(self.cfg.eot_samples):
                x_adv = (x + delta).clamp(0.0, 1.0)

                # Apply random transform (differentiable)
                x_t, _ = apply_transforms_2d(x_adv, self.transform_cfg)

                # Classifier expects BHWC
                logits = self.classifier(x_t.permute(0, 2, 3, 1))  # (1, C)

                if self.cfg.target_class is not None:
                    target = torch.tensor([self.cfg.target_class], device=self.device)
                    loss = F.cross_entropy(logits, target)
                    total_loss = total_loss - loss  # minimize CE ↔ maximize target
                else:
                    pred = logits.argmax(dim=1)
                    target = pred  # maximize CE on current prediction
                    loss = F.cross_entropy(logits, target)
                    total_loss = total_loss + loss

            total_loss = total_loss / self.cfg.eot_samples
            total_loss.backward()

            with torch.no_grad():
                # Gradient sign update
                grad_sign = delta.grad.sign()
                delta.data = delta.data - self.cfg.step_size * grad_sign

                # Project back into L-inf ball
                delta.data = delta.data.clamp(-self.cfg.epsilon, self.cfg.epsilon)

                # Also ensure x+delta stays in [0,1]
                delta.data = (x + delta.data).clamp(0.0, 1.0) - x

            delta.grad.zero_()

            if step % self.cfg.log_every == 0 or step == self.cfg.num_steps - 1:
                with torch.no_grad():
                    x_adv_eval = (x + delta).clamp(0.0, 1.0).permute(0, 2, 3, 1)
                    pred = self.classifier(x_adv_eval).argmax(dim=1).item()
                    history["loss"].append(total_loss.item())
                    history["pred"].append(pred)
                    print(f"  [EOT 2D] step {step:4d}/{self.cfg.num_steps}  "
                          f"loss={total_loss.item():.4f}  pred={pred}")

        adv = (x + delta).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)  # HWC
        return adv.detach(), history
