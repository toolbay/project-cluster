"""JSFlags defender training and inference library."""

from .trainer import train_model
from .inferencer import infer_patch

__all__ = ["train_model", "infer_patch"]
