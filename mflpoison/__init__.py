"""Unified research framework for multimodal federated poisoning experiments.

The original :mod:`fed_multimodal` package remains the FDMM baseline.  This
package provides stable interfaces around generators, attacks, defenses, and
experiment artifacts without breaking legacy checkpoints or entry points.
"""

from .core.types import ClientUpdate, DatasetSpec, MultimodalBatch, SyntheticBatch

__all__ = [
    "ClientUpdate",
    "DatasetSpec",
    "MultimodalBatch",
    "SyntheticBatch",
]
