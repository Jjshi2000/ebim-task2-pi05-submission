#!/usr/bin/env python3
"""Call the deployed full-chunk protocol using a recorded real observation."""
import argparse
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app" if (root / "app").is_dir() else root))
import av
import numpy as np
from navigation.config import load_config
from navigation.dataset import RealDataset
from task2_real_runner import PolicyClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--config", default="config/task2.real.yaml")
    ap.add_argument("--output", default="out/pi05_smoke.json")
    args = ap.parse_args()
    dataset = RealDataset(args.dataset)
    table = dataset.table(args.episode)
    images = {}
    for camera in ("head", "wrist_left", "wrist_right"):
        path = dataset.root / dataset.info["video_path"].format(
            episode_index=args.episode, episode_chunk=args.episode // dataset.info["chunks_size"],
            video_key="observation.images." + camera)
        with av.open(str(path)) as container:
            images[camera] = next(container.decode(video=0)).to_ndarray(format="rgb24")
    cfg = load_config(args.config)["policy"]
    cfg["request_timeout_s"] = 120
    client = PolicyClient("127.0.0.1", args.port, cfg)
    rows = []
    try:
        for index in range(args.chunks):
            start = time.monotonic()
            actions = np.asarray(client.request((table["observation.state"][0], images, {})))
            row = dict(chunk=index, seconds=time.monotonic() - start, shape=list(actions.shape),
                       finite=bool(np.isfinite(actions).all()), first_action=actions[0].tolist())
            assert actions.shape == (50, 17) and row["finite"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        client.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
