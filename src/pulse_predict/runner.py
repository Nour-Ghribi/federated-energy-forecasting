"""Configuration-driven runs; no implicit changes in data or compute budget."""
from __future__ import annotations
from pathlib import Path
import copy
import logging
import hashlib
import json
import time
import numpy as np
import pandas as pd
from .data import locate_csv,load_series,make_samples,align_regions,training_scales
from .evaluation import metrics,assert_manifest,regional_aggregate
from .models import resolve_device,seed_everything,fit_xgboost,fit_lstm,fit_arima,predict_lstm,save_lstm
from .utils import json_write,run_directory
from . import figures

LOG=logging.getLogger(__name__)


def prepare(cfg:dict,root:Path,out:Path,plots:bool=True):
    samples={}
    for region in cfg['data']['regions']:
        s,audit=load_series(locate_csv(root,region),units=cfg['data']['units'])
        data=s.loc[cfg['data']['start']:].copy()
        data=data[data.index<pd.Timestamp(cfg['data']['end'])]
        smp=make_samples(data,cfg);samples[region]=smp
        json_write(out/f'audit_{region}.json',audit)
    if len(samples)>1:align_regions(samples)
    for region,s in samples.items():
        manifest=pd.concat([pd.DataFrame({'timestamp':s.timestamps(k),'split':k,'region':region}) for k in s.indices],ignore_index=True)
        manifest.to_csv(out/f'manifest_{region}.csv',index=False)
        json_write(out/f'coverage_{region}.json',{'grid_hours':len(s.series),'admissible':{k:len(v) for k,v in s.indices.items()},
                                                'masked_or_insufficient_hours':len(s.series)-sum(map(len,s.indices.values()))})
        if plots:figures.eda(s.series,s.x,out/'figures'/region,cfg['purpose'])
    return samples


def run_one(cfg:dict,root:Path,output:Path,model_name:str,seed:int,with_shap:bool=False)->Path:
    cfg=copy.deepcopy(cfg);cfg['seed']=seed;cfg['model']=model_name
    if model_name in {'fedavg','fedprox'}:cfg['federated']['method']=model_name
    device=resolve_device(cfg['device'])
    seed_everything(seed,device)
    out=run_directory(output,f'{model_name}_s{seed}',cfg)
    LOG.info('Starting %s seed=%s -> %s',model_name,seed,out)
    try:
        samples=prepare(cfg,root,out,plots=False)
        return _train(cfg,samples,out,model_name,seed,device,with_shap)
    except Exception as exc:
        json_write(out/'status.json',{'status':'failed','error':str(exc),'type':type(exc).__name__})
        raise


def _train(cfg,samples,out,model_name,seed,device,with_shap):
    scales=training_scales(samples,cfg['federated']['normalization'])
    frames=[];results=[];model=None
    stats={'training_seconds':0.,'train_samples':0}
    if model_name in {'lstm','fedavg','fedprox'}:
        if model_name=='lstm':
            x=np.concatenate([scales[r].transform(s.windows('train')) for r,s in samples.items()])
            y=np.concatenate([scales[r].transform(s.targets('train')) for r,s in samples.items()])
            vx=np.concatenate([scales[r].transform(s.windows('validation')) for r,s in samples.items()])
            vy=np.concatenate([scales[r].transform(s.targets('validation')) for r,s in samples.items()])
            val_std=np.concatenate([np.full(len(s.indices['validation']),scales[r].std,dtype=np.float32) for r,s in samples.items()])
            model,hist,stats=fit_lstm(x,y,vx,vy,cfg['lstm'],seed,device,validation_std=val_std)
            pd.DataFrame(hist).to_csv(out/'loss.csv',index=False);figures.losses(hist,out/'figures',cfg['purpose'])
        else:
            from .federated import fit_federated
            cfg['federated']['method']=model_name
            model,stats=fit_federated(samples,scales,cfg,seed,device,out)
        # Saving moves model to CPU; restore requested device for inference.
        save_lstm(out/'model.pt',model,scales,cfg['lstm']['sequence_length']);model.to(device)
    for region,s in samples.items():
        infer_start=time.perf_counter();local_stats=stats.copy()
        if model_name=='naive':pred=s.tabular('test')['lag_1'].to_numpy()
        elif model_name=='seasonal_daily':pred=s.tabular('test')['lag_24'].to_numpy()
        elif model_name=='seasonal_weekly':pred=s.tabular('test')['lag_168'].to_numpy()
        elif model_name=='arima':
            pred,local_stats,arima=fit_arima(s,cfg['arima']);arima.save(out/f'{region}_arima.pkl')
        elif model_name=='xgboost':
            model,local_stats=fit_xgboost(s,cfg['xgboost'],seed,device)
            model.save_model(out/f'{region}_xgboost.ubj')
            json_write(out/f'{region}_features.json',{'columns':list(s.x.columns),'lags':cfg['data']['lags'],'rolling':cfg['data']['rolling']})
            pd.Series(model.feature_importances_,index=s.x.columns,name='gain_importance').sort_values(ascending=False).to_csv(out/f'{region}_feature_importance.csv')
            infer_start=time.perf_counter();pred=model.predict(s.tabular('test'))
            if with_shap:
                shap_start=time.perf_counter()
                from .explainability import explain
                explain(model,s.tabular('train'),s.tabular('test'),out/'shap'/region,cfg['shap'],cfg['purpose'])
                local_stats['shap_seconds']=time.perf_counter()-shap_start
        elif model_name in {'lstm','fedavg','fedprox'}:
            pred=scales[region].inverse(predict_lstm(model,scales[region].transform(s.windows('test')),device))
        else:raise ValueError(f'Unknown model {model_name}')
        if model_name!='arima':local_stats['inference_seconds']=time.perf_counter()-infer_start-local_stats.get('shap_seconds',0.)
        frame=pd.DataFrame({'timestamp':s.timestamps('test'),'region':region,'actual_mw':s.targets('test'),'prediction_mw':pred})
        assert_manifest(s.timestamps('test'),frame)
        frame.to_csv(out/f'predictions_{region}.csv',index=False)
        figures.predictions(frame,out/'figures'/region,cfg['purpose'])
        row={'model':model_name,'region':region,'seed':seed,'profile':cfg['profile'],'purpose':cfg['purpose'],
             'partition':cfg['federated']['partition'],'normalization':cfg['federated']['normalization'],
             'mu':cfg['federated']['mu'] if model_name=='fedprox' else 0.,
             'sequence_length':cfg['lstm']['sequence_length'],'hidden_size':cfg['lstm']['hidden_size'],
             'test_samples':len(frame),**metrics(frame.actual_mw.to_numpy(),frame.prediction_mw.to_numpy())}
        implementation=json.loads((out/'environment.json').read_text(encoding='utf-8'))['package_source_sha256']
        row['implementation_id']=implementation[:16]
        protocol=copy.deepcopy(cfg);protocol['implementation_id']=implementation
        for key in ('seed','seeds','experiment'):protocol.pop(key,None)
        row['protocol_id']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
        row['manifest_id']=hashlib.sha256(s.timestamps('test').asi8.tobytes()).hexdigest()[:16]
        json_write(out/f'{region}_training.json',local_stats)
        row.update({k:v for k,v in local_stats.items() if isinstance(v,(int,float,str))})
        frames.append(frame);results.append(row)
        if model_name in {'fedavg','fedprox'} and len(samples)==1:
            from .federated import partition_indices
            parts=partition_indices(len(frame),cfg['federated']['clients'],cfg['federated']['partition'],cfg['federated']['partition_seed'])
            pd.DataFrame([{'client':i,'split':'test','samples':len(p),**metrics(frame.actual_mw.to_numpy()[p],frame.prediction_mw.to_numpy()[p])} for i,p in enumerate(parts)]).to_csv(out/'client_test_metrics.csv',index=False)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    json_write(out/'run_cost.json',{'training_seconds':results[0]['training_seconds'] if model_name in {'lstm','fedavg','fedprox'} else sum(r['training_seconds'] for r in results),
                                   'scope':'whole run; shared neural training counted once'})
    json_write(out/'metrics.json',results);json_write(out/'regional_aggregate.json',regional_aggregate(frames))
    json_write(out/'status.json',{'status':'completed','purpose':cfg['purpose']})
    LOG.info('Completed %s',out)
    return out


def summarize(output:Path)->None:
    allrows=[]
    for p in output.glob('*/metrics.csv'):
        f=pd.read_csv(p);f['artifact']=p.parent.name;allrows.append(f)
    if not allrows:raise ValueError('No completed metric artifacts')
    frame=pd.concat(allrows,ignore_index=True);frame.to_csv(output/'all_metrics.csv',index=False)
    keys=['purpose','profile','model','region','partition','normalization','mu','sequence_length','hidden_size','protocol_id','manifest_id']
    # Re-runs must not masquerade as additional independent seeds.
    frame=frame.drop_duplicates(keys+['seed'],keep='last')
    summary=frame.groupby(keys,dropna=False)[['mae','rmse','smape','training_seconds']].agg(['mean','std','count'])
    summary.to_csv(output/'summary.csv')
    costs=frame.drop_duplicates('artifact')[['artifact','model','seed','profile','purpose','protocol_id','training_seconds']].copy()
    for i,row in costs.iterrows():
        p=output/row.artifact/'run_cost.json'
        if p.exists():costs.loc[i,'training_seconds']=json.loads(p.read_text(encoding='utf-8'))['training_seconds']
    costs.to_csv(output/'run_costs.csv',index=False)
    costs.groupby(['purpose','profile','model','protocol_id']).training_seconds.agg(['mean','std','sum','count']).to_csv(output/'cost_summary.csv')
    for (purpose,profile,region,partition,norm,length,hidden,manifest),group in frame.groupby(['purpose','profile','region','partition','normalization','sequence_length','hidden_size','manifest_id']):
        label=f'{purpose}; {region}; {partition}; normalization={norm}; L={length}; H={hidden}'
        safe=f'{profile}_{region}_{partition}_{norm}_L{length}_H{hidden}_{manifest}'
        figures.comparison(group,output/'figures'/safe,label)
    figures.federated_rounds(output,'Validation trajectories; consult run configurations before comparison')
    figures.heterogeneity(frame,output/'figures')
    report=['# Computed results','', 'Generated solely from completed artifacts. Smoke results are functional checks, not research conclusions.','',
            '|Model|Region|Profile|Seed|MAE MW|RMSE MW|Train s|Artifact|','|---|---|---|---:|---:|---:|---:|---|']
    for _,r in frame.iterrows():report.append(f"|{r.model}|{r.region}|{r.profile}|{r.seed}|{r.mae:.3f}|{r.rmse:.3f}|{r.training_seconds:.3f}|{r.artifact}|")
    (output/'computed_findings.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
