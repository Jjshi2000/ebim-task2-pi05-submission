#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--process-pid", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            os.kill(args.process_pid, 0)
        except ProcessLookupError as exc:
            raise SystemExit("inference process exited before opening its port") from exc
        try:
            with socket.create_connection((args.host, args.port), timeout=0.5):
                print(f"inference ready at {args.host}:{args.port}", flush=True)
                return
        except OSError:
            time.sleep(1.0)
    raise SystemExit(f"timed out waiting for inference at {args.host}:{args.port}")


if __name__ == "__main__":
    main()

