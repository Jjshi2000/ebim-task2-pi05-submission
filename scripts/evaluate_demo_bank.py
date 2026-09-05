#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app" if (root / "app").is_dir() else root))
import cv2
import numpy as np

from navigation.config import load_config
from navigation.dataset import RealDataset
from navigation.demo_bank import DemoBank


def perturbations(rgb):
    h, w = rgb.shape[:2]
    for fraction in (0.0, 0.01, 0.03, 0.06, 0.12):
        H = np.float32([[1, 0, w * fraction], [0, 1, 0]])
        yield f"translation_{fraction}", cv2.warpAffine(rgb, H, (w, h))
    for scale in (0.90, 0.97, 1.03, 1.10, 1.25):
        H = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        yield f"scale_{scale}", cv2.warpAffine(rgb, H, (w, h))
    for angle in (2, 5, 12):
        H = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        yield f"rotation_{angle}", cv2.warpAffine(rgb, H, (w, h))
    for gain, offset in ((0.8, 0), (1.1, 15), (0.8, 25)):
        yield f"photometric_{gain}_{offset}", np.clip(rgb.astype(float) * gain + offset, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 55])
    if ok:
        yield "jpeg_55", cv2.cvtColor(cv2.imdecode(jpg, 1), cv2.COLOR_BGR2RGB)
    yield "blank", np.zeros_like(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="config/task2.real.yaml")
    ap.add_argument("--output", default="out/demo_bank_evaluation.json")
    ap.add_argument("--starts-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cv2.setNumThreads(2)
    bank = DemoBank(cfg["navigation"]["visual"]["bank_path"], cfg["navigation"]["visual"])
    dataset = RealDataset(args.dataset)
    rows, temporal, perturb = [], [], []
    reasons = Counter()
    for number, episode in enumerate(dataset.episodes):
        eid = int(episode["episode_index"])
        for meta, rgb in dataset.start_samples(eid, cfg["bank"]["window_s"], cfg["bank"]["frames_per_episode"]):
            e = bank.evaluate(rgb)
            r = e.registration
            row = {**meta, "ready": e.ready, "score": e.score, "reference_id": e.reference_id,
                   "valid": bool(r and r.valid), "reason": e.details.get("reason", r.reason if r else "missing")}
            reasons[row["reason"]] += 1
            rows.append(row)
        if not args.starts_only:
            for second, (rgb, _) in dataset.frames(eid, cfg["bank"]["temporal_seconds"]).items():
                e = bank.evaluate(rgb)
                temporal.append(dict(episode_index=eid, seconds=second, ready=e.ready, score=e.score))
        if number % 30 == 0:
            print(f"evaluated {number + 1}/{len(dataset.episodes)}", flush=True)
    for rmeta in bank.metadata["references"]:
        rgb = dataset.frames(rmeta["episode_index"], [rmeta["timestamp"]])[rmeta["timestamp"]][0]
        for kind, changed in perturbations(rgb):
            e = bank.evaluate(changed, rmeta["reference_id"])
            perturb.append(dict(reference_id=e.reference_id, kind=kind, ready=e.ready, score=e.score,
                                valid=e.registration.valid, reason=e.registration.reason))
    accepted = sum(r["ready"] for r in rows)
    out = dict(episodes=len(dataset.episodes), start_frames=len(rows), accepted=accepted,
               acceptance_fraction=accepted / len(rows), reasons=dict(reasons),
               false_rejects=[r for r in rows if not r["ready"]], starts=rows,
               temporal=temporal, perturbations=perturb,
               limitation="in-distribution replay, not independent site or physical navigation validation")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("episodes", "start_frames", "accepted", "acceptance_fraction", "reasons")}, indent=2))


if __name__ == "__main__":
    main()
