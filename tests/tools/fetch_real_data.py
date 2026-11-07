"""Compatibility wrapper. The maintained fetcher is scripts/fetch_real_data.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.fetch_real_data import main

if __name__ == "__main__":
    main()
