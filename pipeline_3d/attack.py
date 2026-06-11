"""
3D EOT (Expectation Over Transformation) adversarial attack.

Optimizes the texture map of a 3D mesh so that the object is misclassified
from arbitrary viewpoints.

Reference: Athalye et al. 2017 - "Synthesizing Robust Adversarial Examples"

Gradient flow:
    loss  ←  classifier  ←  rendered_image  ←  F.grid_sample  ←  texture_map
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from pipeline_3d.transforms import TransformConfig, sample_transforms


@dataclass
class Attack3DConfig:
    # Target class index (None = untargeted)
    target_class: Optional[int] = None

    # Texture constraint
    # None  → only clamp texture to [0, 1]
    # float → additionally enforce L-inf ball of this radius around original texture
    epsilon: Optional[float] = None

    step_size: float = 0.01
    num_steps: int = 200

    # EOT: number of random viewpoints per gradient step
    eot_samples: int = 20

    log_every: int = 20


class EOTAttack3D:
    """
    PGD-based adversarial texture optimization with EOT over 3D viewpoints.

    Optimizes:
        texture = argmax  E_{t ~ T}[ loss(f(render(mesh_t, texture)), y_target) ]
        s.t.     texture in [0, 1]   (and optionally ||texture - texture_orig||_inf <= eps)

    Args:
        classifier:     callable (B, H, W, 3) → (B, num_classes) logits.
        renderer:       SoftwareTexturedMeshRenderer (or PyTorch3D variant).
                        renderer.texture_map must be an nn.Parameter.
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

    def attack(self) -> Tuple[torch.Tensor, dict]:
        """
        Optimize renderer.texture_map in-place.

        Returns:
            best_texture:  (1, H, W, 3) optimized texture tensor
            history:       dict with 'loss' and 'pred' lists
        """
        renderer = self.renderer
        cfg = self.cfg

        # Save original texture for L-inf constraint (if epsilon given)
        tex_orig = renderer.texture_map.data.clone() if cfg.epsilon is not None else None

        renderer.texture_map.requires_grad_(True)

        history = {"loss": [], "pred": []}

        for step in range(cfg.num_steps):
            if renderer.texture_map.grad is not None:
                renderer.texture_map.grad.zero_()

            step_loss_accum = 0.0

            # Accumulate gradients over EOT samples
            for _ in range(cfg.eot_samples):
                R, T, bg = sample_transforms(1, self.transform_cfg, self.device)
                images, _ = renderer.render_with_background(R, T, bg)  # (1, H, W, 3)

                logits = self.classifier(images)  # (1, num_classes)

                if cfg.target_class is not None:
                    target = torch.tensor([cfg.target_class], device=self.device)
                    loss = -F.cross_entropy(logits, target)   # maximize target class prob
                else:
                    pred = logits.argmax(dim=1).detach()
                    loss = F.cross_entropy(logits, pred)       # maximize CE on current pred

                (loss / cfg.eot_samples).backward()
                step_loss_accum += loss.item()

            step_loss = step_loss_accum / cfg.eot_samples

            # PGD step
            with torch.no_grad():
                renderer.texture_map.data -= cfg.step_size * renderer.texture_map.grad.sign()

                # L-inf projection around original texture (optional)
                if cfg.epsilon is not None:
                    renderer.texture_map.data.clamp_(
                        tex_orig - cfg.epsilon,
                        tex_orig + cfg.epsilon,
                    )

                # Always clamp to valid RGB range
                renderer.texture_map.data.clamp_(0.0, 1.0)

            # Logging
            if step % cfg.log_every == 0 or step == cfg.num_steps - 1:
                with torch.no_grad():
                    R_eval, T_eval, bg_eval = sample_transforms(4, self.transform_cfg, self.device)
                    imgs_eval, _ = renderer.render_with_background(R_eval, T_eval, bg_eval)
                    preds = self.classifier(imgs_eval).argmax(dim=1).tolist()

                history["loss"].append(step_loss)
                history["pred"].append(preds)
                print(f"  [EOT 3D] step {step:4d}/{cfg.num_steps}  "
                      f"loss={step_loss:.4f}  preds={preds}")

        return renderer.texture_map.detach().clone(), history
