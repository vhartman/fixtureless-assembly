#!/usr/bin/env python3
"""Preferred spelling for cube process-wrench scaling (modules or --piecewise)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.run_cube_process_wrench_gravity import main


if __name__ == "__main__":
  raise SystemExit(main())
