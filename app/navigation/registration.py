from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Registration


@dataclass
class Features:
    gray: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    global_descriptor: np.ndarray


class Registrar:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        requested = cfg["feature_type"].upper()
        self.feature_type = requested
        if requested == "SIFT" and hasattr(cv2, "SIFT_create"):
            self.extractor = cv2.SIFT_create(nfeatures=cfg["max_features"], contrastThreshold=cfg["sift_contrast_threshold"])
            self.norm = cv2.NORM_L2
        elif requested in ("SIFT", "AKAZE"):
            self.feature_type = "AKAZE"
            self.extractor = cv2.AKAZE_create()
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError(f"unsupported feature type {requested}")
        self.matcher = cv2.BFMatcher(self.norm)

    def extract(self, rgb: np.ndarray) -> Features:
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
                rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16):
            raise ValueError("expected nonempty HWC uint8 RGB image")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.cfg["width"], self.cfg["height"]), interpolation=cv2.INTER_AREA)
        keypoints, desc = self.extractor.detectAndCompute(gray, None)
        points = np.array([p.pt for p in keypoints], np.float32).reshape(-1, 2)
        if desc is None:
            dtype = np.float32 if self.norm == cv2.NORM_L2 else np.uint8
            desc = np.empty((0, self.extractor.descriptorSize()), dtype)
        small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = (small - small.mean()) / max(float(small.std()), 1.0)
        dx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
        glob = np.concatenate([small.ravel(), dx.ravel(), dy.ravel()])
        glob /= max(float(np.linalg.norm(glob)), 1e-6)
        return Features(gray, points, desc, glob)

    def match(self, current: Features, reference: Features, reference_id: int) -> Registration:
        r = Registration(reference_id, number_of_keypoints=len(current.points))
        cfg = self.cfg
        r.global_distance = float(np.linalg.norm(current.global_descriptor - reference.global_descriptor))

        def invalid(reason):
            r.reason = reason
            return r

        if min(len(current.descriptors), len(reference.descriptors)) < cfg["min_good_matches"]:
            return invalid("too few features")
        pairs = self.matcher.knnMatch(current.descriptors, reference.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg["ratio_test"] * p[1].distance]
        # Repeated texture must not vote multiple times for the same reference feature.
        unique = {}
        for m in sorted(good, key=lambda m: m.distance):
            unique.setdefault(m.trainIdx, m)
        good = list(unique.values())
        r.good_matches = len(good)
        if len(good) < cfg["min_good_matches"]:
            return invalid("too few unique matches")
        src = np.array([current.points[m.queryIdx] for m in good], np.float32)
        dst = np.array([reference.points[m.trainIdx] for m in good], np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg["ransac_threshold"], maxIters=3000, confidence=0.995)
        if H is None or mask is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10:
            return invalid("missing or non-finite homography")
        r.H = H = H / H[2, 2]
        inliers = mask.ravel().astype(bool)
        r.inliers = int(inliers.sum())
        r.inlier_ratio = float(inliers.mean())
        if r.inliers < cfg["min_inliers"] or r.inlier_ratio < cfg["min_inlier_ratio"]:
            return invalid("insufficient RANSAC consensus")
        projected = cv2.perspectiveTransform(src[inliers][None], H)[0]
        r.reprojection_error = float(np.median(np.linalg.norm(projected - dst[inliers], axis=1)))
        if r.reprojection_error > cfg["max_reprojection_error"]:
            return invalid("reprojection error")
        w, h = cfg["width"], cfg["height"]
        area = float(w * h)
        r.spread = min(abs(cv2.contourArea(cv2.convexHull(p[inliers]))) / area for p in (src, dst))
        if r.spread < cfg["min_spread"]:
            return invalid("inliers concentrated in a small region")
        unit = np.diag([w, h, 1.0])
        normalized = np.linalg.inv(unit) @ H @ unit
        if np.linalg.cond(normalized) > cfg["max_condition"]:
            return invalid("ill-conditioned homography")
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        denom = np.c_[corners, np.ones(4)] @ H[2]
        if np.any(denom <= 1e-6) or denom.max() / denom.min() > cfg["max_projective_ratio"]:
            return invalid("projective pole or excessive perspective")
        r.corners = out = cv2.perspectiveTransform(corners[None], H)[0]
        if not np.isfinite(out).all() or not cv2.isContourConvex(out):
            return invalid("nonconvex or non-finite corners")
        signed_area = cv2.contourArea(out, oriented=True)
        if signed_area <= 0:
            return invalid("reflected homography")
        r.scale = float(np.sqrt(signed_area / area))
        r.rotation = float(np.arctan2(out[1, 1] - out[0, 1], out[1, 0] - out[0, 0]))
        r.translation = tuple(float(v) for v in (out.mean(axis=0) - corners.mean(axis=0)))
        r.geometry = ((out - corners) / np.array([w, h])).ravel().astype(float)
        if not cfg["min_scale"] <= r.scale <= cfg["max_scale"] or abs(r.rotation) > cfg["max_rotation_rad"]:
            return invalid("scale or rotation outside limits")
        if np.max(np.abs(out / np.array([w, h]))) > cfg["max_corner_extent"]:
            return invalid("corners outside plausible extent")
        intersection, _ = cv2.intersectConvexConvex(corners, out.astype(np.float32))
        r.overlap = float(intersection / max(area, signed_area))
        if r.overlap < cfg["min_overlap"]:
            return invalid("insufficient overlap")
        warped = cv2.warpPerspective(current.gray, H, (w, h))
        valid = cv2.warpPerspective(np.ones_like(current.gray), H, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
        a = cv2.Sobel(warped.astype(np.float32), cv2.CV_32F, 1, 0)[valid]
        b = cv2.Sobel(reference.gray.astype(np.float32), cv2.CV_32F, 1, 0)[valid]
        r.similarity = float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6))
        r.valid, r.reason = True, "valid"
        return r
