"""Capture vendor versions before pip can replace the Jetson CUDA runtime."""
from pathlib import Path
import torch
import torchvision

Path("/tmp/ngc-constraints.txt").write_text(
    f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n")
