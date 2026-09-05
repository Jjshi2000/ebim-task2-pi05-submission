from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .registration import Features, Registrar
from .types import Evaluation
from .visual_score import score_registration


def medoids(descriptors: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    distances = np.linalg.norm(descriptors[:, None] - descriptors[None, :], axis=2)
    # Balanced divisive groups avoid allocating most references to rare outliers.
    groups = [np.arange(len(descriptors))]
    while len(groups) < min(k, len(descriptors)):
        index = max(range(len(groups)), key=lambda j: len(groups[j]))
        members = groups.pop(index)
        local = distances[np.ix_(members, members)]
        a, b = np.unravel_index(local.argmax(), local.shape)
        projection = local[:, a] - local[:, b]
        ordered = members[np.argsort(projection, kind="stable")]
        groups.extend(np.array_split(ordered, 2))
    chosen, labels = [], np.empty(len(descriptors), dtype=int)
    for j, members in enumerate(groups):
        chosen.append(int(members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))]))
        labels[members] = j
    return chosen, labels


class DemoBank:
    def __init__(self, path: str | Path, cfg: dict):
        path = Path(path)
        self.metadata = json.loads((path / "metadata.json").read_text())
        if self.metadata["format_version"] != 1:
            raise ValueError("unsupported reference bank format")
        saved = self.metadata["visual_config"]
        # Registration settings define the measured acceptance distribution.
        for key in saved:
            if key not in ("bank_path", "shortlist_k") and saved[key] != cfg[key]:
                raise ValueError(f"bank configuration mismatch: {key}; rebuild the bank")
        self.registrar = Registrar(cfg)
        if self.registrar.feature_type != self.metadata["feature_type"]:
            raise ValueError("feature implementation differs from reference bank")
        blob = path / "references.npz"
        if hashlib.sha256(blob.read_bytes()).hexdigest() != self.metadata["references_sha256"]:
            raise ValueError("reference bank checksum mismatch")
        with np.load(blob, allow_pickle=False) as arrays:
            self.references = [Features(arrays[f"gray_{i}"], arrays[f"points_{i}"],
                                        arrays[f"descriptors_{i}"], arrays[f"global_{i}"])
                               for i in range(len(self.metadata["references"]))]
        if not any(r["statistics"].get("calibrated") for r in self.metadata["references"]):
            raise ValueError("no reference has enough independent calibration episodes")
        self.cfg = cfg

    def evaluate(self, rgb: np.ndarray, reference_id: int | None = None) -> Evaluation:
        features = self.registrar.extract(rgb)
        if reference_id is None:
            distances = [np.linalg.norm(features.global_descriptor - r.global_descriptor) for r in self.references]
            ids = np.argsort(distances, kind="stable")[:self.cfg["shortlist_k"]]
        else:
            ids = [reference_id]
        results = []
        for i in ids:
            r = self.registrar.match(features, self.references[i], int(i))
            stats = self.metadata["references"][i]["statistics"]
            results.append(score_registration(r, stats, self.cfg))
        return max(results, key=lambda e: (e.ready, e.score))
