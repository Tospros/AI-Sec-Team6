"""
ImageNet classifier wrapper (InceptionV3 by default).

Provides a unified interface for getting class logits/probabilities
from rendered or real images, matching the setup in the paper.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models
from typing import Optional


# ImageNet class indices used in the paper's demo
IMAGENET_CLASSES = {
    "turtle": 35,     # "leatherback turtle" (35) or "mud turtle" (37)
    "rifle":  764,    # "rifle"
    "baseball": 429,  # "baseball"
}


class InceptionV3Classifier(nn.Module):
    """
    Pretrained InceptionV3 on ImageNet.

    Input images are expected as (B, H, W, 3) tensors in [0, 1]
    (renderer output format). Handles the channel-first conversion
    and InceptionV3's expected normalization internally.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        # Disable gradient tracking for classifier parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Normalization constants from ImageNet (torchvision standard)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        )

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, H, W, 3) [0,1] → (B, 3, H, W) normalized for InceptionV3.
        Also resizes to 299x299 if needed.
        """
        # HWC → CHW, scale to [0,1]
        x = images.permute(0, 3, 1, 2).float()  # (B, 3, H, W)
        x = x.to(self.device)

        # Resize to InceptionV3's expected 299×299
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Normalize
        x = (x - self.mean) / self.std
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, H, W, 3) in [0, 1]

        Returns:
            logits: (B, 1000) — raw (un-softmaxed) class scores
        """
        x = self.preprocess(images)
        # InceptionV3 returns InceptionOutputs(logits, aux_logits) during training;
        # in eval mode it returns just the logits tensor.
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Returns (B,) predicted class indices."""
        with torch.no_grad():
            logits = self(images)
        return logits.argmax(dim=1)

    def top5(self, images: torch.Tensor):
        """Returns top-5 (class_idx, probability) pairs for each image."""
        with torch.no_grad():
            logits = self(images)
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_idx = probs.topk(5, dim=1)
        return top5_idx, top5_probs
