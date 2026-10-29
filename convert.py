"""Kept so `python convert.py ...` from 1.x keeps working. Prefer the `precision-convert` command."""

import sys

from precision_converter.cli import main

if __name__ == "__main__":
    sys.exit(main())
