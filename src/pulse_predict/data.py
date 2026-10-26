"""Auditable ingestion and strictly chronological forecasting samples."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def locate_csv(root: Path, region: str) -> Path:
    for p in (root / "data" / "raw" / f"{region}_hourly.csv", root / f"{region}_hourly.csv"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing {region}_hourly.csv in {root}/data/raw or {root}")


def load_series(path: Path, target: str | None = None, units: str = "MW") -> tuple[pd.Series, dict]:
    """Keep civil source hours; mask ALL occurrences of ambiguous timestamps.

    No inference of UTC, no interpolation, no clipping. MW is a declared source
    unit, not something a numeric CSV can independently establish.
    """
    if units != "MW":
        raise ValueError("Explicit MW source units are required")
    raw = pd.read_csv(path)
    dates = [c for c in raw if c.lower() in {"datetime", "timestamp", "date"}]
    if len(dates) != 1:
        raise ValueError(f"Expected one timestamp column, found {dates}")
    choices = [c for c in raw if c.endswith("_MW")]
    if target is None:
        if len(choices) != 1:
            raise ValueError("Select target explicitly for a multi-region/unknown schema")
        target = choices[0]
    if target not in raw or target == dates[0]:
        raise ValueError(f"Invalid target {target}")
    dt = pd.to_datetime(raw[dates[0]], errors="raise")
    if dt.dt.tz is not None:
        raise ValueError("This loader expects naive source civil hours; configure an explicit timezone adapter")
    if not ((dt.dt.minute == 0) & (dt.dt.second == 0)).all():
        raise ValueError("Non-hour-aligned timestamps")
    values = pd.to_numeric(raw[target], errors="raise")
    if np.isinf(values).any():
        raise ValueError("Infinite target values")
    duplicate = dt.duplicated(keep=False)
    valid = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dt), name=target)[~duplicate.to_numpy()].sort_index()
    grid = pd.date_range(dt.min(), dt.max(), freq="h", name="timestamp")
    series = valid.reindex(grid)
    unique = pd.DatetimeIndex(dt.unique()).sort_values()
    irregular = unique.to_series().diff().dropna()
    q1, q3 = values.quantile([.25, .75])
    audit = {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": list(raw.columns), "target": target, "units": units,
        "units_basis": "declared PJM/Kaggle metadata; not inferred from values",
        "timezone_policy": "source civil hours; no UTC mapping; ambiguous hours masked",
        "rows": len(raw), "start": str(grid.min()), "end": str(grid.max()),
        "duplicate_rows": int(duplicate.sum()), "duplicate_excess": int(dt.duplicated().sum()),
        "ambiguous_timestamps": sorted(dt[duplicate].astype(str).unique().tolist()),
        "irregular_intervals": {str(k): int(v) for k, v in irregular[irregular != pd.Timedelta(hours=1)].value_counts().items()},
        "missing_grid_hours": int(len(grid.difference(pd.DatetimeIndex(dt)))),
        "missing_timestamps": grid.difference(pd.DatetimeIndex(dt)).astype(str).tolist(),
        "missing_values_raw": int(values.isna().sum()), "masked_grid_hours": int(series.isna().sum()),
        "negative_values": int((values < 0).sum()),
        "iqr_outliers_unmodified": int(((values < q1-3*(q3-q1)) | (values > q3+3*(q3-q1))).sum()),
        "statistics": values.describe().to_dict(),
    }
    if (values < 0).any():
        raise ValueError("Negative load: inspect source before modeling")
    return series, audit


def features(series: pd.Series, lags: list[int], rolling: list[int]) -> pd.DataFrame:
    if not lags or min(lags) < 1 or any(w < 2 for w in rolling):
        raise ValueError("Lags must be positive and rolling windows >=2")
    idx = series.index
    x = pd.DataFrame(index=idx)
    for name, value in {"hour": idx.hour, "weekday": idx.dayofweek, "month": idx.month,
                        "weekend": (idx.dayofweek >= 5).astype(int)}.items():
        x[name] = value
    for name, value, period in [("hour", idx.hour, 24), ("weekday", idx.dayofweek, 7)]:
        x[f"{name}_sin"] = np.sin(2*np.pi*value/period)
        x[f"{name}_cos"] = np.cos(2*np.pi*value/period)
    for lag in lags:
        x[f"lag_{lag}"] = series.shift(lag)
    for w in rolling:
        history = series.shift(1).rolling(w, min_periods=w)
        x[f"mean_{w}"] = history.mean()
        x[f"std_{w}"] = history.std()
    return x


@dataclass
class SlidingWindowDataset:
    """Map-style dataset accepted by PyTorch DataLoader without eager copies."""
    values: np.ndarray
    positions: np.ndarray
    sequence_length: int

    def __post_init__(self) -> None:
        if self.sequence_length < 1 or len(self.positions)==0:
            raise ValueError('Nonempty positions and a positive sequence length are required')
        if min(self.positions)<self.sequence_length or max(self.positions)>=len(self.values):
            raise ValueError('Window positions outside observed series')

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        i=int(self.positions[index])
        return self.values[i-self.sequence_length:i,None].astype(np.float32), np.float32(self.values[i])


@dataclass
class Samples:
    series: pd.Series
    x: pd.DataFrame
    indices: dict[str, np.ndarray]
    sequence_length: int

    def windows(self, split: str) -> np.ndarray:
        """At target t, input contains observed loads [t-L, ..., t-1]."""
        y = self.series.to_numpy(dtype=np.float32)
        dataset=SlidingWindowDataset(y,self.indices[split],self.sequence_length)
        return np.stack([dataset[i][0] for i in range(len(dataset))])

    def targets(self, split: str) -> np.ndarray:
        return self.series.iloc[self.indices[split]].to_numpy(dtype=np.float32)

    def tabular(self, split: str) -> pd.DataFrame:
        return self.x.iloc[self.indices[split]]

    def timestamps(self, split: str) -> pd.DatetimeIndex:
        return self.series.index[self.indices[split]]


def make_samples(series: pd.Series, cfg: dict, sequence_length: int | None = None) -> Samples:
    d = cfg["data"]
    length = sequence_length or cfg["lstm"]["sequence_length"]
    x = features(series, d["lags"], d["rolling"])
    # Use maximum sequence length in the comparison manifest, including sensitivity experiments.
    support = max(length, d["comparison_history"], max(d["lags"]), max(d["rolling"]))
    valid = series.notna() & x.notna().all(axis=1)
    valid &= series.shift(1).rolling(support, min_periods=support).count().eq(support)
    start, val, test, end = [pd.Timestamp(d[k]) for k in ("start", "validation_start", "test_start", "end")]
    if not start < val < test < end:
        raise ValueError("Require start < validation_start < test_start < end")
    # The start boundary defines data access, including historical window inputs.
    valid &= series.index >= start + pd.Timedelta(hours=support)
    bounds = {"train": (start, val), "validation": (val, test), "test": (test, end)}
    indices = {k: np.flatnonzero(valid & (series.index >= a) & (series.index < b)) for k, (a, b) in bounds.items()}
    if any(len(v) == 0 for v in indices.values()):
        raise ValueError(f"Empty split after continuity checks: { {k:len(v) for k,v in indices.items()} }")
    return Samples(series, x, indices, length)


def align_regions(samples: dict[str, Samples]) -> None:
    """Fix a shared manifest before fitting, never intersect model predictions."""
    for split in ("train", "validation", "test"):
        common = None
        for s in samples.values():
            common = s.timestamps(split) if common is None else common.intersection(s.timestamps(split))
        if common.empty:
            raise ValueError("No common regional timestamps")
        for s in samples.values():
            s.indices[split] = s.series.index.get_indexer(common)


@dataclass
class Scale:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> Scale:
        if not np.isfinite(values).all() or values.size == 0:
            raise ValueError("Scale requires finite training data")
        return cls(float(values.mean()), max(float(values.std()), 1e-8))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values-self.mean)/self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values*self.std+self.mean


def training_scales(samples: dict[str, Samples], mode: str) -> dict[str, Scale]:
    if mode not in {"common", "local"}:
        raise ValueError("normalization must be common or local")
    # Unique observed training hours, rather than duplicated overlapping windows.
    data = {}
    for region, s in samples.items():
        pos = np.unique(np.concatenate([np.arange(i-s.sequence_length, i+1) for i in s.indices["train"]]))
        data[region] = s.series.iloc[pos].to_numpy()
    if mode == "common":
        scale = Scale.fit(np.concatenate(list(data.values())))
        return {r: scale for r in samples}
    return {r: Scale.fit(v) for r, v in data.items()}
