#!/usr/bin/env python3
"""Read-only preflight using exactly the deployed topic/freshness checks."""
import sys
from task2_real_runner import main

if __name__ == "__main__":
    raise SystemExit(main(["--preflight", *sys.argv[1:]]))
