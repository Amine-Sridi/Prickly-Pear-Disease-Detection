"""Seed helper and environment capture (library versions, GPU, platform)."""
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import random
import subprocess
import sys

TRACKED_PACKAGES = (
    "torch", "torchvision", "ultralytics", "diffusers", "transformers", "accelerate", "peft",
    "huggingface-hub", "safetensors", "numpy", "pillow", "opencv-python", "opencv-python-headless",
)

# Pinned in requirements-colab.txt. torch is deliberately NOT pinned (Colab's preinstalled
# CUDA build is kept); its version is recorded in every run JSON instead.
PINNED = {
    "ultralytics": "8.4.183",
    "diffusers": "0.41.0",
    "transformers": "5.20.2",
    "accelerate": "1.15.0",
    "peft": "0.21.2",
}


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. Ultralytics also seeds itself from its `seed` argument."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_versions(packages=TRACKED_PACKAGES) -> dict:
    out = {}
    for p in packages:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def check_pins() -> dict:
    """Return {package: (wanted, installed)} for every pin that does not match."""
    have = package_versions(PINNED)
    return {p: (want, have.get(p)) for p, want in PINNED.items() if have.get(p) != want}


def gpu_info() -> dict:
    info = {"cuda_available": False}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if info["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            info.update(name=p.name, total_memory_gb=round(p.total_memory / 1024**3, 2),
                        capability=f"{p.major}.{p.minor}", device_count=torch.cuda.device_count())
    except ImportError:
        info["torch"] = None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        info["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = None
    return info


def collect_env() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "colab": "google.colab" in sys.modules or os.path.exists("/content"),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "pins_mismatch": check_pins(),
        "gpu": gpu_info(),
    }


def pip_freeze() -> str:
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return r.stdout
