"""Tile-level metrics.

Every label is one whole-tile box, so the detector is effectively classifying
tiles. Besides Ultralytics' detection metrics we therefore report a simple,
threshold-free readout: the class of the single highest-confidence box per
tile ("none" if the model outputs no box at all, counted as a miss).
"""
from __future__ import annotations

from collections import Counter

CLASSES = ("Healthy", "Diseased")
NONE = "none"


def tile_confusion(y_true: list[str], y_pred: list[str], classes=CLASSES) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    pred_labels = (*classes, NONE)
    counts = Counter(zip(y_true, y_pred))
    unknown = {t for t in y_true if t not in classes} | {p for p in y_pred if p not in pred_labels}
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    confusion = {f"true_{t}": {f"pred_{p}": counts[(t, p)] for p in pred_labels} for t in classes}
    per_class = {}
    for c in classes:
        tp = counts[(c, c)]
        n_true = sum(counts[(c, p)] for p in pred_labels)
        n_pred = sum(counts[(t, c)] for t in classes)
        per_class[c] = {
            "n_true": n_true,
            "n_pred": n_pred,
            "tp": tp,
            "recall": tp / n_true if n_true else None,
            "precision": tp / n_pred if n_pred else None,  # None = class never predicted (not 0, not 1)
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "rule": "class of the highest-confidence box per tile (conf>=0.001); 'none' = no box = miss",
        "n": len(y_true),
        "confusion": confusion,
        "per_class": per_class,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "accuracy": sum(counts[(c, c)] for c in classes) / len(y_true) if y_true else None,
    }
