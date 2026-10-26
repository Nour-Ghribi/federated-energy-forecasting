"""Metrics in original units with exact evaluation-manifest enforcement."""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 1 or not y.size or not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("Expected equally shaped nonempty finite 1D observations and predictions")
    err = pred-y
    denom = np.abs(y)+np.abs(pred)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.square(err).mean())),
            "smape": float(100*np.divide(2*np.abs(err), denom, out=np.zeros_like(err), where=denom!=0).mean())}


def assert_manifest(expected: pd.DatetimeIndex, predictions: pd.DataFrame) -> None:
    actual = pd.DatetimeIndex(pd.to_datetime(predictions["timestamp"]))
    if not expected.equals(actual) or actual.has_duplicates:
        raise ValueError("Prediction timestamps do not exactly match the predefined evaluation manifest")
    if not np.isfinite(predictions[["actual_mw", "prediction_mw"]].to_numpy()).all():
        raise ValueError("Missing/nonfinite predictions; evaluation cannot shrink silently")


def regional_aggregate(frames: list[pd.DataFrame]) -> dict:
    per = [metrics(f.actual_mw.to_numpy(), f.prediction_mw.to_numpy()) for f in frames]
    pooled = pd.concat(frames, ignore_index=True)
    return {"macro": {k: float(np.mean([v[k] for v in per])) for k in per[0]},
            "pooled": metrics(pooled.actual_mw.to_numpy(), pooled.prediction_mw.to_numpy())}
