#!/usr/bin/env python3
"""Compatibility entry point for per-client scenario generator training."""

import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mflpoison.runner.__main__ import main


if __name__ == "__main__":
    warnings.warn(
        "standalone centralized generator training is deprecated; "
        "pass a complete scenario with --config",
        DeprecationWarning,
    )
    raise SystemExit(main())
