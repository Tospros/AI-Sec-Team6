"""
3D EOT (Expectation Over Transformation) adversarial-mask attack.

Optimizes a single screen-space mask (delta) that, when overlaid on the 2D
projection of a 3D mesh rendered from an arbitrary viewpoint, fools the
classifier. The mask is "looked through": we render the mesh, add the mask to
that projection, and classify the result.

    delta = argmax  E_{view ~ V}[ loss(f(render(mesh, view) + delta), y) ]
    s.t.   ||delta||_inf <= epsilon   and   render + delta in [0, 1]

Reference: Athalye et al. 2017 - "Synthesizing Robust Adversarial Examples"

Note: delta is added *after* rendering, so the render is constant w.r.t. delta.
We render under no_grad and only backprop into the mask — the attack therefore
works even when the underlying rasterizer is non-differentiable.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass

from pipeline_3d.transforms import TransformConfig, sample_transforms


@dataclass
class Attack3DConfig:
    # Target class index (None = untargeted)
    target_class: Optional[int] = None

    # Class indices to suppress (e.g. all turtle classes). When set and no
    # target_class is given, the objective drives the total probability mass on
    # these classes toward zero — i.e. "make the classifier stop saying turtle".
    protected_classes: Optional[List[int]] = None

    # L-inf bound on the screen-space mask
    epsilon: float = 0.05

    step_size: float = 0.005
    num_steps: int = 200

    # EOT: number of random viewpoints rendered per gradient step
    eot_samples: int = 20

    log_every: int = 20


class EOTMaskAttack3D:
    """
    PGD optimization of a universal screen-space adversarial mask, robust over
    random 3D viewpoints (EOT).

    Args:
        classifier:     callable (B, H, W, 3) -> (B, num_classes) logits.
        renderer:       textured-mesh renderer exposing render_with_background
                        and `image_size`.
        transform_cfg:  distribution of random 3D camera poses.
        attack_cfg:     attack hyperparameters.
        device:         torch device.
    """

    def __init__(
        self,
        classifier: Callable,
        renderer,
        transform_cfg: Optional[TransformConfig] = None,
        attack_cfg: Optional[Attack3DConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.classifier = classifier
        self.renderer = renderer
        self.transform_cfg = transform_cfg or TransformConfig()
        self.cfg = attack_cfg or Attack3DConfig()
        self.device = device or torch.device("cpu")

    def _render_views(self, n: int) -> torch.Tensor:
        """Render n random viewpoints. Constant w.r.t. the mask, so no grad."""
        with torch.no_grad():
            R, T, bg = sample_transforms(n, self.transform_cfg, self.device)
            images, _ = self.renderer.render_with_background(R, T, bg)
        return images.detach()  # (n, H, W, 3)

    def _eval_fool(self, n: int, delta: torch.Tensor) -> Tuple[float, list, Optional[float]]:
        """
        Render fresh views, apply the mask, and report how well it works.

        Returns (rate, adv_preds, protected_prob):
            rate            success metric (see below)
            adv_preds       per-view top-1 class after the mask
            protected_prob  mean prob mass on protected_classes after the mask
                            (None when protected_classes is not set)
        """
        cfg = self.cfg
        with torch.no_grad():
            imgs = self._render_views(n)
            clean_pred = self.classifier(imgs).argmax(dim=1)
            adv = (imgs + delta).clamp(0.0, 1.0)
            adv_logits = self.classifier(adv)
            adv_pred = adv_logits.argmax(dim=1)

            prot_prob = None
            if cfg.target_class is not None:
                rate = (adv_pred == cfg.target_class).float().mean().item()
            elif cfg.protected_classes is not None:
                prot = torch.tensor(cfg.protected_classes, device=self.device)
                prot_prob = torch.softmax(adv_logits, dim=1)[:, prot].sum(dim=1).mean().item()
                # success = view no longer classified as any protected class
                is_prot = (adv_pred.unsqueeze(1) == prot.unsqueeze(0)).any(dim=1)
                rate = (~is_prot).float().mean().item()
            else:
                rate = (adv_pred != clean_pred).float().mean().item()
        return rate, adv_pred.tolist(), prot_prob

    def attack(self) -> Tuple[torch.Tensor, dict]:
        """
        Optimize the screen-space mask.

        Returns:
            delta:    (1, H, W, 3) optimized mask (values in [-eps, eps])
            history:  dict with 'loss' and 'fool_rate' lists
        """
        cfg = self.cfg
        H = W = self.renderer.image_size

        prot = (torch.tensor(cfg.protected_classes, device=self.device)
                if cfg.protected_classes is not None else None)

        # Initialize the mask uniformly in [-eps, eps]
        delta = torch.empty(1, H, W, 3, device=self.device).uniform_(-cfg.epsilon, cfg.epsilon)
        delta.requires_grad_(True)

        history = {"step": [], "loss": [], "fool_rate": [], "protected_prob": []}

        for step in range(cfg.num_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            # Render a fresh batch of viewpoints (the EOT expectation samples).
            # Fresh every step => the mask is optimized over the whole mesh's
            # viewpoint distribution, not a fixed set of projections.
            imgs = self._render_views(cfg.eot_samples)  # (B, H, W, 3), constant w.r.t. delta

            adv = (imgs + delta).clamp(0.0, 1.0)        # "look through the mask"
            logits = self.classifier(adv)               # (B, num_classes)

            if cfg.target_class is not None:
                target = torch.full(
                    (imgs.shape[0],), cfg.target_class, device=self.device, dtype=torch.long
                )
                loss = F.cross_entropy(logits, target)   # minimize -> push toward target
                sign = -1.0
            elif prot is not None:
                p = torch.softmax(logits, dim=1)[:, prot].sum(dim=1)
                loss = -(1.0 - p + 1e-6).log().mean()    # minimize -> suppress protected mass
                sign = -1.0
            else:
                with torch.no_grad():
                    clean_pred = self.classifier(imgs).argmax(dim=1)
                loss = F.cross_entropy(logits, clean_pred)  # maximize -> push away from clean
                sign = +1.0

            loss.backward()

            # PGD step + projection onto the L-inf ball
            with torch.no_grad():
                delta.data += sign * cfg.step_size * delta.grad.sign()
                delta.data.clamp_(-cfg.epsilon, cfg.epsilon)

            # Logging
            if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                rate, preds, prot_prob = self._eval_fool(
                    min(8, max(4, cfg.eot_samples)), delta.detach())
                history["step"].append(step)
                history["loss"].append(loss.item())
                history["fool_rate"].append(rate)
                history["protected_prob"].append(prot_prob)
                if cfg.target_class is not None:
                    tag = "success"
                elif prot is not None:
                    tag = "de-turtled"
                else:
                    tag = "fooled"
                extra = f"  turtle_p={prot_prob:.0%}" if prot_prob is not None else ""
                print(f"  [EOT 3D mask] step {step:4d}/{cfg.num_steps}  "
                      f"loss={loss.item():.4f}  {tag}={rate:.0%}{extra}  preds={preds}")

        return delta.detach(), history


# Backwards-compatible alias (the attack is now mask-based, not texture-based).
EOTAttack3D = EOTMaskAttack3D
