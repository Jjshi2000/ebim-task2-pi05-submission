#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app" if (root / "app").is_dir() else root))
import cv2
import numpy as np

from navigation.config import load_config
from navigation.dataset import RealDataset
from navigation.demo_bank import medoids
from navigation.registration import Registrar
from navigation.visual_score import calibrate


def main():
    parser = argparse.ArgumentParser(description="Build an offline bank from every released real episode")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="config/task2.real.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bc, vc = cfg["bank"], cfg["navigation"]["visual"]
    cv2.setNumThreads(2)
    cv2.setRNGSeed(bc["seed"])
    dataset = RealDataset(args.dataset)
    if len(dataset.episodes) != bc["expected_episodes"]:
        raise ValueError("dataset does not have the configured episode count")
    registrar = Registrar(vc)
    feats, records = [], []
    poses = {"left": [], "right": []}
    for number, episode in enumerate(dataset.episodes):
        eid = int(episode["episode_index"])
        table = dataset.table(eid)
        for side in poses:
            poses[side].append(table[f"observation.state.franka_robot_{side}_measured_joint_states"][0])
        for meta, rgb in dataset.start_samples(eid, bc["window_s"], bc["frames_per_episode"]):
            f = registrar.extract(rgb)
            records.append({**meta, "image_dimensions": list(rgb.shape), "keypoints": len(f.points)})
            feats.append(f)
        if number % 20 == 0:
            print(f"extracted {number + 1}/{len(dataset.episodes)} episodes", flush=True)
    chosen, labels = medoids(np.stack([f.global_descriptor for f in feats]), bc["clusters"])
    arrays, refs = {}, []
    for j, chosen_index in enumerate(chosen):
        f = feats[chosen_index]
        for key, value in (("gray", f.gray), ("points", f.points), ("descriptors", f.descriptors), ("global", f.global_descriptor)):
            arrays[f"{key}_{j}"] = value
        members = [int(i) for i in range(len(feats))
                   if records[i]["episode_index"] != records[chosen_index]["episode_index"]]
        matches = [registrar.match(feats[i], f, j) for i in members]
        stats = calibrate(matches, [records[i]["episode_index"] for i in members], vc)
        refs.append({"reference_id": j, **records[chosen_index], "cluster_size": int((labels == j).sum()),
                     "statistics": stats})
        print(f"reference {j}: {stats}", flush=True)
    output = Path(args.output or vc["bank_path"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "references.npz", **arrays)
    for i, record in enumerate(records):
        record["cluster_id"] = int(labels[i])
    visual_config = {k: v for k, v in vc.items() if k != "bank_path"}
    metadata = dict(format_version=1, feature_type=registrar.feature_type, visual_config=visual_config,
                    dataset={k: bc[k] for k in ("dataset_repo", "dataset_revision", "dataset_subdir")},
                    episodes=len(dataset.episodes), samples=records, references=refs,
                    start_pose={side: np.median(values, axis=0).tolist() for side, values in poses.items()},
                    start_pose_p05={side: np.quantile(values, .05, axis=0).tolist() for side, values in poses.items()},
                    start_pose_p95={side: np.quantile(values, .95, axis=0).tolist() for side, values in poses.items()},
                    references_sha256=hashlib.sha256((output / "references.npz").read_bytes()).hexdigest())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not any(r["statistics"].get("calibrated") for r in refs):
        raise RuntimeError("no calibrated references; the saved bank is unusable")
    print(f"bank ready: {output}", flush=True)


if __name__ == "__main__":
    main()
