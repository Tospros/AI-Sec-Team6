"""
Generate the figures used in report_2d_explained/REPORT.md.

This script *replicates* the EOT-PGD loop from pipeline_2d/attack.py line-for-line
so we can capture the intermediate tensors the supervisor asked about:

  * attack.py L105  ->  x_t  : the image actually fed to the classifier,
                                i.e. (x + delta) after a random 2D transform.
  * attack.py L122  ->  delta.grad : the gradient of the EOT loss w.r.t. the
                                perturbation, whose .sign() drives the PGD step.

Run from the repo root with the project venv:
    venv/bin/python report_2d_explained/generate_figures.py
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.classifier import InceptionV3Classifier
from pipeline_2d.transforms import (
    Transform2DConfig,
    sample_affine_matrix,
    apply_spatial_transform,
    apply_color_jitter,
    apply_noise,
    apply_transforms_2d,
)
from pipeline_2d.attack import AttackConfig

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

# ── ImageNet labels (for readable titles) ──────────────────────────────
from torchvision import models as _tvm
CATEGORIES = _tvm.Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]


def lbl(i):
    return f"{i}:{CATEGORIES[i][:18]}"


def chw_to_hwc(t):
    """(C,H,W) tensor -> (H,W,C) numpy, clipped to [0,1]."""
    return t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def load_image(path, size=299):
    img = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(img)).float() / 255.0  # (H,W,3) in [0,1]


# ── Setup ──────────────────────────────────────────────────────────────
IMAGE_PATH = os.path.join(os.path.dirname(HERE), "images.jpg")
image = load_image(IMAGE_PATH)                       # (H,W,3)
clf = InceptionV3Classifier(device=device)

with torch.no_grad():
    top5_idx, top5_p = clf.top5(image.unsqueeze(0).to(device))
    pred0 = int(top5_idx[0, 0])
print("Original top-5:", [(int(i), CATEGORIES[int(i)], float(p))
                          for i, p in zip(top5_idx[0], top5_p[0])])

# Same config family as demo_2d_attack.py, shrunk for CPU.
tcfg = Transform2DConfig(rotation_deg=15.0, scale_min=0.9, scale_max=1.1,
                         gaussian_noise_std=0.02)
TARGET = 764  # rifle
acfg = AttackConfig(target_class=TARGET, epsilon=0.05, step_size=0.05 / 20,
                    num_steps=25, eot_samples=12)

# ── Replicate attack.py loop ───────────────────────────────────────────
x = image.unsqueeze(0).to(device).permute(0, 3, 1, 2)        # (1,C,H,W)  == attack.py L86
delta = torch.empty_like(x).uniform_(-acfg.epsilon, acfg.epsilon)
delta.requires_grad_(True)

facts = {}
history = {"loss": [], "target_prob": [], "pred": []}
captured_xt = None       # snapshot of x_t at L105 (one EOT sample)
captured_grad = None     # snapshot of delta.grad at L122 (last step)

for step in range(acfg.num_steps):
    total_loss = torch.tensor(0.0, device=device)
    for j in range(acfg.eot_samples):
        x_adv = (x + delta).clamp(0.0, 1.0)                  # L99
        x_t, _ = apply_transforms_2d(x_adv, tcfg)            # L102
        # ---- attack.py L105: the image fed to the classifier is x_t ----
        logits = clf(x_t.permute(0, 2, 3, 1))                # L105
        target = torch.tensor([TARGET], device=device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        total_loss = total_loss - loss                       # L110 (targeted)
        if step == acfg.num_steps - 1 and j == 0:
            captured_xt_adv = x_adv.detach().clone()
    total_loss = total_loss / acfg.eot_samples
    total_loss.backward()                                    # L118

    # ---- attack.py L122: delta.grad / its sign drives the step ----
    if step == acfg.num_steps - 1:
        captured_grad = delta.grad.detach().clone()

    with torch.no_grad():
        grad_sign = delta.grad.sign()                        # L122
        delta.data = delta.data - acfg.step_size * grad_sign # L123
        delta.data = delta.data.clamp(-acfg.epsilon, acfg.epsilon)
        delta.data = (x + delta.data).clamp(0.0, 1.0) - x
    delta.grad.zero_()

    with torch.no_grad():
        x_eval = (x + delta).clamp(0.0, 1.0).permute(0, 2, 3, 1)
        lg = clf(x_eval)
        p = torch.softmax(lg, 1)
        history["loss"].append(total_loss.item())
        history["target_prob"].append(float(p[0, TARGET]))
        history["pred"].append(int(lg.argmax(1)))
    print(f"step {step:2d} loss={total_loss.item():.3f} "
          f"P(target)={history['target_prob'][-1]:.3f} pred={history['pred'][-1]}")

adv = (x + delta).clamp(0.0, 1.0)

# ── FIGURE 02: what the classifier sees at L105 (x_t) ──────────────────
torch.manual_seed(1)
x_adv_now = (x + delta).clamp(0.0, 1.0)
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
ax[0, 0].imshow(chw_to_hwc(x_adv_now[0]))
ax[0, 0].set_title("x_adv = clamp(x+delta)\n(BEFORE transform)")
for k in range(7):
    xt, theta = apply_transforms_2d(x_adv_now, tcfg)
    r, c = divmod(k + 1, 4)
    ax[r, c].imshow(chw_to_hwc(xt[0]))
    ax[r, c].set_title(f"x_t sample #{k+1}\n(fed to classifier @L105)")
for a in ax.ravel():
    a.axis("off")
fig.suptitle("attack.py L105 — the classifier input x_t is x_adv after a random EOT transform",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_eot_samples_xt.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ── FIGURE 03: transform stages in isolation ───────────────────────────
torch.manual_seed(3)
theta = sample_affine_matrix(1, tcfg, device)
stage0 = x_adv_now
stage1 = apply_spatial_transform(stage0, theta)
stage2 = apply_color_jitter(stage1.clone(), tcfg)
stage3 = apply_noise(stage2.clone(), tcfg.gaussian_noise_std)
stages = [("0. x_adv (input)", stage0),
          ("1. + affine (grid_sample)\nrot/scale/translate", stage1),
          ("2. + color jitter\nbrightness/contrast", stage2),
          ("3. + gaussian noise\n= x_t @L102", stage3)]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for a, (t, s) in zip(ax, stages):
    a.imshow(chw_to_hwc(s[0]))
    a.set_title(t, fontsize=11)
    a.axis("off")
fig.suptitle("How apply_transforms_2d composes the transform: spatial -> color -> noise",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_transform_stages.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ── FIGURE 04: delta.grad at L122 ──────────────────────────────────────
g = captured_grad[0]                       # (C,H,W)
g_mag = g.abs().sum(0)                      # (H,W) total magnitude
gabs_max = float(g.abs().max())
facts["grad_min"] = float(g.min())
facts["grad_max"] = float(g.max())
facts["grad_mean"] = float(g.mean())
facts["grad_std"] = float(g.std())
facts["grad_abs_max"] = gabs_max
facts["grad_frac_positive_sign"] = float((g > 0).float().mean())


def norm01(t):
    t = t - t.min()
    return t / (t.max() + 1e-12)


# raw grad mapped to [0,1] around 0.5 (so sign is visible)
g_centered = (g / (gabs_max + 1e-12)) * 0.5 + 0.5            # tiny amplitude -> ~grey
g_amp10 = (g / (gabs_max + 1e-12) * 10).clamp(-1, 1) * 0.5 + 0.5
g_sign = delta.grad if False else captured_grad.sign()[0]    # the actual L122 quantity
g_sign_vis = g_sign * 0.5 + 0.5

fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
ax[0].imshow(chw_to_hwc(g_centered))
ax[0].set_title("delta.grad (raw, normalized)\nlooks ~flat grey: amplitude tiny")
ax[1].imshow(chw_to_hwc(g_amp10))
ax[1].set_title("delta.grad x10 amplitude\n(structure now visible)")
ax[2].imshow(g_mag.detach().cpu().numpy(), cmap="inferno")
ax[2].set_title("|grad| magnitude heatmap\n(where the loss is sensitive)")
ax[3].imshow(chw_to_hwc(g_sign_vis))
ax[3].set_title("grad.sign() — the value\nactually used @L122/L123")
for a in ax:
    a.axis("off")
fig.suptitle("attack.py L122 — delta.grad: tiny per-pixel amplitude, but rich spatial structure",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_delta_grad.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ── FIGURE 05: attack result ───────────────────────────────────────────
with torch.no_grad():
    adv_top5_idx, adv_top5_p = clf.top5(adv.permute(0, 2, 3, 1))
adv_pred = int(adv_top5_idx[0, 0])
pert = (adv - x)[0]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
ax[0].imshow(chw_to_hwc(x[0]))
ax[0].set_title(f"original\npred {lbl(pred0)}")
ax[1].imshow(chw_to_hwc(adv[0]))
ax[1].set_title(f"adversarial\npred {lbl(adv_pred)}")
ax[2].imshow(((pert * 10) + 0.5).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy())
ax[2].set_title("perturbation x10")
for a in ax:
    a.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "05_attack_result.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ── FIGURE 06: progress curves ─────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history["loss"]); ax[0].set_title("EOT loss (-CE to target)"); ax[0].set_xlabel("step")
ax[1].plot(history["target_prob"]); ax[1].set_title("P(target=rifle) on clean adv"); ax[1].set_xlabel("step")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "06_progress.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

facts.update({
    "orig_pred": pred0, "orig_label": CATEGORIES[pred0],
    "adv_pred": adv_pred, "adv_label": CATEGORIES[adv_pred],
    "target": TARGET, "target_label": CATEGORIES[TARGET],
    "final_target_prob": history["target_prob"][-1],
    "epsilon": acfg.epsilon, "step_size": acfg.step_size,
    "num_steps": acfg.num_steps, "eot_samples": acfg.eot_samples,
    "xt_min": float(stage3.min()), "xt_max": float(stage3.max()),
})
with open(os.path.join(OUT, "facts.json"), "w") as f:
    json.dump(facts, f, indent=2)
print("\nFACTS:", json.dumps(facts, indent=2))
print("Figures written to", OUT)
