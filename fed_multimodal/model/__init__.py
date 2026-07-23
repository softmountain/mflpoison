"""Multimodal model compatibility exports used by the UCF101 adapter."""

from .mm_models import (
    ECGClassifier,
    HARClassifier,
    ImageTextClassifier,
    MMActionClassifier,
    SERClassifier,
)

__all__ = [
    "MMActionClassifier",
    "SERClassifier",
    "ImageTextClassifier",
    "HARClassifier",
    "ECGClassifier",
]
