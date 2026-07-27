#!/usr/bin/env python3
"""Deprecated wrapper for temporal-adaptive checkpoint evaluation."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fed_multimodal.legacy_evaluation.temporal_adaptive import main


if __name__ == "__main__":
    raise SystemExit(main())
