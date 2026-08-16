#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Task 2 PI0.5 checkpoint")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    args = parser.parse_args()

    target = Path(args.local_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=target,
        token=os.environ.get("HF_TOKEN"),
    )

    missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise SystemExit(f"downloaded checkpoint is incomplete; missing: {', '.join(missing)}")
    print(f"checkpoint ready: {target}", flush=True)


if __name__ == "__main__":
    main()
