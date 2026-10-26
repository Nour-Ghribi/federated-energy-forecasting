"""Checkpoint inference on previously unseen, schema-validated hourly data."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_series,features
from .models import load_lstm,predict_lstm


def infer(checkpoint:Path,input_csv:Path,region:str)->pd.DataFrame:
    series,_=load_series(input_csv)
    if checkpoint.suffix=='.pt':
        model,scales,length=load_lstm(checkpoint)
        if region not in scales:raise ValueError(f'Checkpoint has no scale for region {region}')
        # Inputs may include historical context. Predict target times including
        # the next unseen hour; no target observation is required for inference.
        series=series.reindex(pd.date_range(series.index.min(),series.index.max()+pd.Timedelta(hours=1),freq='h'))
        pos=np.flatnonzero(series.shift(1).rolling(length,min_periods=length).count().eq(length))
        if not len(pos):raise ValueError('Insufficient contiguous observed history')
        windows=np.stack([series.iloc[i-length:i].to_numpy(dtype=np.float32)[:,None] for i in pos])
        pred=scales[region].inverse(predict_lstm(model,scales[region].transform(windows),'cpu'))
    elif checkpoint.suffix=='.ubj':
        import json
        from xgboost import XGBRegressor
        model=XGBRegressor();model.load_model(checkpoint)
        meta=json.loads((checkpoint.parent/f'{region}_features.json').read_text(encoding='utf-8'))
        series=series.reindex(pd.date_range(series.index.min(),series.index.max()+pd.Timedelta(hours=1),freq='h'))
        x=features(series,meta['lags'],meta['rolling']);pos=np.flatnonzero(x.notna().all(axis=1))
        if not len(pos):raise ValueError('Insufficient history for feature construction')
        pred=model.predict(x.iloc[pos][meta['columns']])
    else:raise ValueError('Expected LSTM .pt or XGBoost .ubj checkpoint')
    return pd.DataFrame({'timestamp':series.index[pos],'region':region,'prediction_mw':pred})
