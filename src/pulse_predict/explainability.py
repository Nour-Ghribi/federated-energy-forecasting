from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from .figures import save


def explain(model,train:pd.DataFrame,test:pd.DataFrame,out:Path,cfg:dict,label:str)->None:
    out.mkdir(parents=True,exist_ok=True)
    n=min(len(test),cfg['samples']);b=min(len(train),cfg['background'])
    selected=np.unique(np.linspace(0,len(test)-1,n,dtype=int))
    background=train.iloc[np.unique(np.linspace(0,len(train)-1,b,dtype=int))]
    sample=test.iloc[selected]
    # Match XGBoost's best_iteration prediction range after early stopping.
    booster=model.get_booster()[:model.best_iteration+1]
    explainer=shap.TreeExplainer(booster,data=background,feature_perturbation='interventional')
    values=explainer(sample,check_additivity=True)
    np.savez_compressed(out/'shap_values.npz',values=values.values,base_values=values.base_values,
                        data=sample.to_numpy(),feature_names=np.asarray(sample.columns),timestamps=sample.index.astype(str).to_numpy(dtype=str))
    pd.Series(np.abs(values.values).mean(axis=0),index=sample.columns,name='mean_abs_shap_mw').sort_values(ascending=False).to_csv(out/'importance.csv')
    shap.plots.beeswarm(values,show=False,max_display=15);save(plt.gcf(),out/'summary',label)
    shap.plots.bar(values,show=False,max_display=15);save(plt.gcf(),out/'importance',label)
    shap.plots.waterfall(values[0],show=False,max_display=12);save(plt.gcf(),out/'local',label)
    top=int(np.argmax(np.abs(values.values).mean(axis=0)))
    shap.plots.scatter(values[:,top],show=False);save(plt.gcf(),out/'dependence',label)
    periods=pd.DatetimeIndex(sample.index).to_period('M').astype(str)
    f=pd.DataFrame(np.abs(values.values),columns=sample.columns);f['period']=periods
    f.groupby('period').mean().to_csv(out/'importance_by_period.csv')
