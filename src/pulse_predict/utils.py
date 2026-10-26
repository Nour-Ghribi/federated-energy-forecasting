from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import os
import subprocess
import numpy as np
import yaml


def merge(a: dict, b: dict) -> dict:
    import copy
    result=copy.deepcopy(a)
    for k,v in b.items():
        result[k]=merge(result[k],v) if isinstance(v,dict) and isinstance(result.get(k),dict) else copy.deepcopy(v)
    return result


def read_config(path: Path, profile: str = "smoke") -> dict:
    raw=yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg={}
    for name in raw.pop("defaults",[]):
        cfg=merge(cfg,yaml.safe_load((path.parent/name).read_text(encoding="utf-8")))
    cfg=merge(cfg,raw)
    cfg=merge(cfg,yaml.safe_load((path.parent/'profiles'/f'{profile}.yaml').read_text(encoding='utf-8')))
    if cfg['lstm']['input_size'] != 1:
        raise ValueError('This baseline LSTM accepts load-only windows: input_size must be 1')
    for key in ('epochs','patience','batch_size','sequence_length'):
        if cfg['lstm'][key]<1: raise ValueError(f'{key} must be positive')
    if cfg['federated']['rounds']<1 or cfg['federated']['local_epochs']<1:
        raise ValueError('Federated rounds and local epochs must be positive')
    return cfg


def json_write(path: Path, data: object) -> None:
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=convert,allow_nan=False),encoding='utf-8')


def environment() -> dict:
    import torch
    packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','torch','flwr','xgboost','shap','statsmodels','scikit-learn')}
    source=hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob('*.py')):
        source.update(file.name.encode());source.update(file.read_bytes())
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    return {'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
            'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),
            'torch_threads':torch.get_num_threads(),'packages':packages,
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_available':torch.cuda.is_available(), 'cuda_version':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'git_commit':commit,
            'package_source_sha256':source.hexdigest()}


def run_directory(root: Path, label: str, cfg: dict) -> Path:
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:8]
    name=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{label}_{digest}"
    out=root/name; out.mkdir(parents=True,exist_ok=False)
    json_write(out/'config.json',cfg); json_write(out/'environment.json',environment())
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    return out
