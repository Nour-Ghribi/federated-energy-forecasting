"""Centralized models and shared federated local optimization."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import random
import time
import os
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import Samples, Scale


def seed_everything(seed: int, device: str = "cpu") -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(request: str) -> str:
    if request not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda or auto")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU profile requested but CUDA is unavailable; select cpu-light explicitly")
    return "cuda" if request == "auto" and torch.cuda.is_available() else ("cpu" if request == "auto" else request)


class LSTMForecaster(nn.Module):
    """[B,L,input_size] -> recurrent [B,L,H] -> last state [B,H] -> [B]."""
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.architecture = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def proximal_penalty(model: nn.Module, reference: list[torch.Tensor], mu: float) -> torch.Tensor:
    if mu < 0:
        raise ValueError("mu must be nonnegative")
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(reference):
        raise ValueError("Reference parameter count mismatch")
    value = torch.zeros((), device=params[0].device)
    for p, ref in zip(params, reference, strict=True):
        if p.shape != ref.shape:
            raise ValueError("Reference parameter shape mismatch")
        value = value + (p-ref.detach().to(p.device)).square().sum()
    return (mu/2)*value


def train_epoch(model: LSTMForecaster, x: np.ndarray, y: np.ndarray, cfg: dict,
                device: str, seed: int, mu: float = 0.0,
                reference: list[torch.Tensor] | None = None,
                optimizer: torch.optim.Optimizer | None = None) -> dict:
    model.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=cfg["batch_size"], shuffle=True, generator=generator, num_workers=0)
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    reference = reference if reference is not None else [p.detach().clone() for p in model.parameters() if p.requires_grad]
    total = 0.0; count = 0; updates = 0
    criterion = nn.MSELoss() if cfg.get("loss", "mse") == "mse" else nn.L1Loss()
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        if mu:
            loss = loss + proximal_penalty(model, reference, mu)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.get("gradient_clip", 1.0))
        optimizer.step()
        total += float(loss.detach())*len(by); count += len(by); updates += 1
    return {"loss": total/count, "examples_processed": count, "updates": updates}


def predict_lstm(model: LSTMForecaster, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval(); outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(torch.from_numpy(x[start:start+batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def new_lstm(cfg: dict, device: str) -> LSTMForecaster:
    return LSTMForecaster(**{k:cfg[k] for k in ("input_size", "hidden_size", "num_layers", "dropout")}).to(device)


def fit_lstm(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
             cfg: dict, seed: int, device: str,
             validation_std: np.ndarray | None = None) -> tuple[LSTMForecaster, list[dict], dict]:
    seed_everything(seed, device); model = new_lstm(cfg, device)
    best = float("inf"); state = None; stale = 0; history = []; updates = 0; examples = 0
    started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    for epoch in range(cfg["epochs"]):
        row = train_epoch(model, x, y, cfg, device, seed+epoch, optimizer=optimizer)
        updates += row["updates"]; examples += row["examples_processed"]
        pred = predict_lstm(model, vx, device)
        val = float(np.square(pred-vy).mean()) if cfg.get("loss","mse")=="mse" else float(np.abs(pred-vy).mean())
        error=(pred-vy)*(validation_std if validation_std is not None else 1.0)
        selection=float(np.sqrt(np.square(error).mean())) if cfg.get('loss','mse')=='mse' else float(np.abs(error).mean())
        history.append({"epoch": epoch+1, "train_loss": row["loss"], "validation_loss": val,
                        "validation_selection":selection})
        if selection < best:
            best=selection; state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
        if stale >= cfg["patience"]:
            break
    if state is None:
        raise RuntimeError("No finite best checkpoint")
    model.load_state_dict(state)
    return model, history, {"training_seconds": time.perf_counter()-started, "epochs_completed": len(history),
                            "updates":updates, "examples_processed":examples, "train_samples":len(y),
                            "selection_metric":('rmse' if cfg.get('loss','mse')=='mse' else 'mae')+('_mw' if validation_std is not None else '_normalized')}


def save_lstm(path: Path, model: LSTMForecaster, scales: dict[str, Scale], length: int) -> None:
    torch.save({"state_dict": model.cpu().state_dict(), "architecture":model.architecture,
                "scales": {k:asdict(v) for k,v in scales.items()}, "sequence_length":length}, path)


def load_lstm(path: Path, device: str = "cpu") -> tuple[LSTMForecaster, dict[str, Scale], int]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model=LSTMForecaster(**payload["architecture"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, {k:Scale(**v) for k,v in payload["scales"].items()}, payload["sequence_length"]


def fit_xgboost(s: Samples, cfg: dict, seed: int, device: str):
    from xgboost import XGBRegressor
    best = None; score = float("inf"); trials = []; started=time.perf_counter()
    for depth in cfg["max_depths"]:
        model=XGBRegressor(n_estimators=cfg["n_estimators"], max_depth=depth,
                           learning_rate=cfg["learning_rate"], subsample=cfg["subsample"],
                           colsample_bytree=1.0, objective="reg:squarederror", tree_method="hist",
                           early_stopping_rounds=cfg["patience"], random_state=seed,
                           n_jobs=2, device=device)
        model.fit(s.tabular("train"),s.targets("train"),
                  eval_set=[(s.tabular("validation"),s.targets("validation"))],verbose=False)
        rmse=float(np.sqrt(np.square(model.predict(s.tabular("validation"))-s.targets("validation")).mean()))
        trials.append({"max_depth":depth,"validation_rmse":rmse,"best_iteration":int(model.best_iteration)})
        if rmse < score:
            best=model;score=rmse
    params={k:('NaN (XGBoost missing-value default)' if isinstance(v,float) and not np.isfinite(v) else v)
            for k,v in best.get_params().items()}
    return best, {"training_seconds":time.perf_counter()-started,"trials":trials,
                  "selected_params":params,"train_samples":len(s.indices["train"])}


def fit_arima(s: Samples, cfg: dict) -> tuple[np.ndarray, dict, object]:
    """Bounded AIC selection, then rolling one-step forecasts with fixed parameters.

    Kalman updates use each newly observed load, including validation/test history;
    no parameter refitting on validation or test, missing hours remain missing.
    """
    from statsmodels.tsa.arima.model import ARIMA
    cutoff=s.timestamps("train").max()
    start=cutoff-np.timedelta64(cfg["training_days"],"D")
    train=s.series.loc[start:cutoff]
    best=None; score=float("inf"); trials=[]; started=time.perf_counter()
    for order in cfg["orders"]:
        fit=ARIMA(train.to_numpy(),order=tuple(order),enforce_stationarity=False,enforce_invertibility=False).fit()
        converged=bool(fit.mle_retvals.get("converged",True))
        trials.append({"order":order,"aic":float(fit.aic),"converged":converged})
        if converged and np.isfinite(fit.aic) and fit.aic<score:
            best=fit; score=fit.aic
    if best is None:
        raise RuntimeError("No converged ARIMA candidate")
    fit_seconds=time.perf_counter()-started
    # Run the complete post-training stream once with frozen parameters. Its
    # one-step filtered predictions depend only on preceding observations.
    end=s.timestamps("test").max()
    future=s.series.loc[cutoff+np.timedelta64(1,"h"):end]
    infer_start=time.perf_counter()
    streamed=best.extend(future.to_numpy())
    forecasts=np.asarray(streamed.get_prediction().predicted_mean)
    positions=future.index.get_indexer(s.timestamps("test"))
    return forecasts[positions], {"training_seconds":fit_seconds,"inference_seconds":time.perf_counter()-infer_start,
                                  "train_samples":int(train.notna().sum()), "training_days":cfg["training_days"],
                                  "trials":trials,"selected_order":list(best.model.order)}, best
