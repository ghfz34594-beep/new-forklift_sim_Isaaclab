"""Train the dual-camera RGB student from teacher-generated action labels."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).with_name("train_approach_bc.py")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
