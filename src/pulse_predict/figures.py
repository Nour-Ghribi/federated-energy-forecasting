"""Artifact-driven, noninteractive research figures (PNG and PDF)."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':120,'savefig.dpi':200,'axes.spines.top':False,
                     'axes.spines.right':False,'font.size':10,'figure.constrained_layout.use':True})


def save(fig,path:Path,title:str=''):
    path.parent.mkdir(parents=True,exist_ok=True)
    if title:fig.suptitle(title,fontsize=10)
    fig.savefig(path.with_suffix('.png'));fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def eda(series:pd.Series,x:pd.DataFrame,out:Path,label:str):
    fig,ax=plt.subplots(figsize=(10,3));series.resample('D').mean().plot(ax=ax,lw=.7)
    ax.set(ylabel='Daily mean load (MW)',xlabel='Source civil time');save(fig,out/'demand',label)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,key,name in zip(axes,[series.index.hour,series.index.dayofweek,series.index.month],['Hour','Weekday (Monday=0)','Month'],strict=True):
        series.groupby(key).mean().plot(ax=ax,marker='o');ax.set(xlabel=name,ylabel='Mean load (MW)')
    save(fig,out/'seasonality',label)
    frame=x.copy();frame['target_mw']=series
    corr=frame.corr();fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90);ax.set_yticks(range(len(corr)),corr.columns)
    fig.colorbar(im,ax=ax,label='Pearson correlation');save(fig,out/'correlation',label)


def predictions(frame:pd.DataFrame,out:Path,label:str):
    t=pd.to_datetime(frame.timestamp);fig,ax=plt.subplots(figsize=(10,3))
    ax.plot(t[:168],frame.actual_mw.iloc[:168],label='Observed',lw=1)
    ax.plot(t[:168],frame.prediction_mw.iloc[:168],label='Predicted',lw=1)
    ax.set(ylabel='Load (MW)',xlabel='Source civil time');ax.legend();save(fig,out/'forecast',label)
    err=frame.prediction_mw-frame.actual_mw;fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].hist(err,bins=40);axes[0].set(xlabel='Prediction − observation (MW)',ylabel='Count')
    axes[1].plot(t,err,lw=.5);axes[1].set(ylabel='Residual (MW)',xlabel='Source civil time')
    save(fig,out/'residuals',label)


def losses(history:list[dict],out:Path,label:str):
    f=pd.DataFrame(history);fig,ax=plt.subplots(figsize=(6,3))
    ax.plot(f.epoch,f.train_loss,label='Training');ax.plot(f.epoch,f.validation_loss,label='Validation')
    ax.set(xlabel='Epoch',ylabel='Normalized loss');ax.legend();save(fig,out/'loss',label)


def comparison(frame:pd.DataFrame,out:Path,label:str):
    if frame.empty:return
    frame=frame.copy()
    frame['model']=frame.apply(lambda r:f'fedprox (mu={r.mu:g})' if r.model=='fedprox' else r.model,axis=1)
    grouped=frame.groupby('model')[['mae','rmse','training_seconds']].agg(['mean','std'])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['mae','rmse'],strict=True):
        ax.bar(grouped.index,grouped[metric]['mean'],yerr=grouped[metric]['std'].fillna(0))
        ax.tick_params(axis='x',rotation=70);ax.set(ylabel=f'{metric.upper()} (MW)')
    save(fig,out/'model_comparison',label)
    fig,ax=plt.subplots(figsize=(8,4))
    for name,row in grouped.iterrows():
        ax.scatter(row['training_seconds']['mean'],row['mae']['mean']);ax.annotate(name,(row['training_seconds']['mean'],row['mae']['mean']),fontsize=7)
    ax.set(xlabel='Mean training duration (s)',ylabel='MAE (MW)');save(fig,out/'accuracy_cost',label)


def federated_rounds(out:Path,label:str):
    groups={}
    for path in out.glob('*/rounds.csv'):
        cfg=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        status=path.parent/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='completed':continue
        fed=cfg['federated'];lstm=cfg['lstm']
        key=(cfg['purpose'],cfg['profile'],'-'.join(cfg['data']['regions']),fed['partition'],fed['normalization'],
             lstm['sequence_length'],lstm['hidden_size'],fed['backend'],fed['rounds'],fed['local_epochs'])
        f=pd.read_csv(path);f['seed']=cfg['seed'];f['method']=cfg['model']+(f"_mu{fed['mu']:g}" if cfg['model']=='fedprox' else '')
        groups.setdefault(key,[]).append(f)
    for key,frames in groups.items():
        f=pd.concat(frames).drop_duplicates(['method','seed','round'],keep='last')
        fig,ax=plt.subplots(figsize=(8,4))
        for method,g in f.groupby('method'):
            stats=g.groupby('round').validation_mae.agg(['mean','std']);std=stats['std'].fillna(0)
            ax.plot(stats.index,stats['mean'],label=method)
            ax.fill_between(stats.index,stats['mean']-std,stats['mean']+std,alpha=.15)
        ax.set(xlabel='Communication round',ylabel='Validation MAE (MW)');ax.legend(fontsize=7)
        name='_'.join(map(str,key))
        save(fig,out/'figures'/f'federated_rounds_{name}',str(key[0])+'; '+str(key[2:5]))


def heterogeneity(frame:pd.DataFrame,out:Path):
    neural=frame[frame.model.isin(['lstm','fedavg','fedprox'])]
    keys=['purpose','profile','region','normalization','sequence_length','hidden_size','mu','manifest_id']
    for values,group in neural.groupby(keys):
        if group.partition.nunique()<2:continue
        table=group.groupby(['partition','model']).mae.mean().unstack('model')
        ax=table.plot.bar(figsize=(8,4),rot=0);ax.set(ylabel='Mean held-out MAE (MW)',xlabel='Simulated partition')
        name='_'.join(str(v) for v in values)
        save(ax.figure,out/f'heterogeneity_{name}','IID-like vs temporal/regional partitions; '+str(values[0]))


def client_diagnostics(profiles:pd.DataFrame,distances:pd.DataFrame,out:Path,label:str):
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for cid,f in profiles.groupby('client'):
        axes[0].plot(f.hour,f.mean_mw,label=f'Client {cid}')
    axes[0].set(xlabel='Source hour',ylabel='Training mean load (MW)');axes[0].legend(fontsize=7)
    count=profiles.client.nunique();matrix=np.zeros((count,count))
    for _,r in distances.iterrows():matrix[int(r.client_a),int(r.client_b)]=matrix[int(r.client_b),int(r.client_a)]=r.wasserstein_normalized
    im=axes[1].imshow(matrix,cmap='viridis');axes[1].set(xlabel='Client',ylabel='Client')
    fig.colorbar(im,ax=axes[1],label='Wasserstein / pooled std')
    save(fig,out/'client_distributions',label)
