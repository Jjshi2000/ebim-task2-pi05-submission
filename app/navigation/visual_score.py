from __future__ import annotations

import numpy as np

from .types import Evaluation, Registration


def geometry_distance(geometry: np.ndarray, stats: dict) -> float:
    z = (geometry - np.asarray(stats["median"])) / np.asarray(stats["spread"])
    return float(np.sqrt(np.mean(z * z)))


def score_registration(r: Registration, stats: dict, cfg: dict) -> Evaluation:
    if not r.valid or r.geometry is None or not stats.get("calibrated"):
        return Evaluation(-1e6, False, r.reference_id, registration=r,
                          details={"reason": r.reason if not r.valid else "uncalibrated reference"})
    distance = geometry_distance(r.geometry, stats)
    quality = np.mean([r.inlier_ratio, r.inliers / (r.inliers + cfg["min_inliers"]),
                       np.exp(-r.reprojection_error), r.overlap,
                       (r.similarity + 1.0) / 2.0])
    score = -distance + cfg["quality_weight"] * float(quality)
    ready = (distance <= stats["distance_limit"] and
             r.inlier_ratio >= stats["inlier_ratio_low"] and
             r.reprojection_error <= stats["reprojection_high"] and
             r.similarity >= stats["similarity_low"])
    return Evaluation(score, bool(ready), r.reference_id, registration=r,
                      details={"geometry_distance": distance, "distance_limit": stats["distance_limit"],
                               "quality": float(quality)})


def calibrate(results: list[Registration], episode_ids: list[int], cfg: dict) -> dict:
    rows = [(r, e) for r, e in zip(results, episode_ids) if r.valid]
    if not rows:
        return {"calibrated": False, "episodes": 0}
    a = np.stack([r.geometry for r, _ in rows])
    center = np.median(a, axis=0)
    mad = np.median(np.abs(a - center), axis=0)
    spread = np.maximum(1.4826 * mad, cfg["geometry_floors"])
    stats = {"median": center.tolist(), "mad": mad.tolist(), "spread": spread.tolist(),
             "p05": np.quantile(a, 0.05, axis=0).tolist(), "p95": np.quantile(a, 0.95, axis=0).tolist(),
             "episodes": len(set(e for _, e in rows)), "registrations": len(rows)}
    d = [geometry_distance(r.geometry, stats) for r, _ in rows]
    # Cross-episode measurements only; a medoid's self-match cannot calibrate a gate.
    stats.update(calibrated=stats["episodes"] >= cfg["min_calibration_episodes"],
                 distance_limit=float(max(np.quantile(d, cfg["acceptance_quantile"]), 1.0) * cfg["acceptance_margin"]),
                 inlier_ratio_low=float(max(cfg["min_inlier_ratio"], np.quantile([r.inlier_ratio for r, _ in rows], 0.05) / cfg["acceptance_margin"])),
                 reprojection_high=float(min(cfg["max_reprojection_error"], max(0.5, np.quantile([r.reprojection_error for r, _ in rows], 0.95)) * cfg["acceptance_margin"])),
                 similarity_low=float(np.quantile([r.similarity for r, _ in rows], 0.05) - 0.10))
    return stats
