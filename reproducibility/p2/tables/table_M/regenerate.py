#!/usr/bin/env python3
"""Regenerate and compare manuscript appendix M: Packing exclusions."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tablelib  # noqa: E402
import comparator  # noqa: E402


def main():
    out = tablelib.build("M")
    result = comparator.run("M", out["tables"], out["gaps"], HERE)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
