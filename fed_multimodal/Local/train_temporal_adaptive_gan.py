#!/usr/bin/env python3
"""Deprecated alias for the partition-safe UCF101 scenario runner."""

import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mflpoison.runner.__main__ import main


if __name__ == "__main__":
    warnings.warn(
        "train_temporal_adaptive_gan.py now runs the unified scenario; "
        "pass --config and optionally --artifact-root",
        DeprecationWarning,
    )
    raise SystemExit(main())
