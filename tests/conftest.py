"""Pytest configuration ensuring path setup and dependency availability."""  # FIX: centralise test bootstrapping

from __future__ import annotations  # FIX: postpone annotations for compatibility

import sys
from pathlib import Path

import pytest

try:  # FIX: guard optional numpy dependency
    import numpy  # noqa: F401  # FIX: imported for availability check
except ImportError:  # FIX: skip suite when numpy missing
    pytest.skip("numpy required for analytics tests", allow_module_level=True)

try:  # FIX: guard optional pandas dependency
    import pandas  # noqa: F401  # FIX: imported for availability check
except ImportError:  # FIX: skip suite when pandas missing
    pytest.skip("pandas required for dataset construction", allow_module_level=True)

try:  # FIX: guard optional torch dependency
    import torch  # noqa: F401  # FIX: imported for availability check
except ImportError:  # FIX: skip suite when torch missing
    pytest.skip("torch required for model components", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]  # FIX: locate repository root for module imports
if str(REPO_ROOT) not in sys.path:  # FIX: avoid duplicate path entries
    sys.path.insert(0, str(REPO_ROOT))  # FIX: expose src package for tests
