# ─────────────────────────────────────────────────────────────────────────────
# Base: PyTorch 2.1.0 + CUDA 11.8 (official image — Python 3.10, numpy, scipy
#       torchvision already installed)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Build arg: set to 1 to compile PyTorch3D from source (~15 min)
ARG PYTORCH3D=0

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        gcc \
        g++ \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# ── PyTorch3D (optional — only for 3D container) ──────────────────────────────
# Compiles from source with CUDA support. Takes 10-20 minutes.
RUN if [ "$PYTORCH3D" = "1" ]; then \
        FORCE_CUDA=1 pip install --no-cache-dir \
            "git+https://github.com/facebookresearch/pytorch3d.git@stable"; \
    fi

# ── Project code ──────────────────────────────────────────────────────────────
COPY models/         models/
COPY pipeline_2d/    pipeline_2d/
COPY pipeline_3d/    pipeline_3d/
COPY demo_2d_attack.py .
COPY demo_3d_attack.py .
COPY demo_3d_pipeline.py .

# Non-interactive matplotlib backend (no display in container)
ENV MPLBACKEND=Agg

# Suppress tokenizer parallelism warning from HuggingFace (if present)
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "--version"]
