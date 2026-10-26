from __future__ import annotations
import argparse
import copy
import logging
from pathlib import Path
from .utils import read_config,run_directory,json_write
from .runner import prepare,run_one,summarize


def experiment_jobs(number:int,cfg:dict):
    e=cfg['experiments']
    if number==1:
        for name in e['baseline']:yield name,copy.deepcopy(cfg),False
    elif number==2:
        for length in e['sequence_lengths']:
            for hidden in e['hidden_sizes']:
                c=copy.deepcopy(cfg);c['lstm'].update(sequence_length=length,hidden_size=hidden)
                yield 'lstm',c,False
    elif number==3:
        for name in ('lstm','fedavg','fedprox'):
            yield name,copy.deepcopy(cfg),False
    elif number==4:
        for partition in e['partitions']:
            for norm in (['common','local'] if partition=='regional' else ['common']):
                c=copy.deepcopy(cfg);c['federated'].update(partition=partition,normalization=norm)
                if partition=='regional':
                    c['data']['regions']=e['regions']
                    # Fixed common regional coverage, never picked from test outcomes.
                    if c['data']['start']<'2013-06-01':c['data']['start']='2013-06-01'
                yield 'lstm',copy.deepcopy(c),False
                yield 'fedavg',copy.deepcopy(c),False
                for mu in e['mu_values']:
                    m=copy.deepcopy(c);m['federated']['mu']=mu
                    yield 'fedprox',m,False
    elif number==5:
        yield 'xgboost',copy.deepcopy(cfg),True
    elif number!=6:raise ValueError('Experiment number must be 1..6')


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description='Federated Energy Forecasting: explainable hourly load prediction')
    parser.add_argument('command',choices=['prepare','train','federated','experiment','summarize','shap','infer','evaluate'])
    parser.add_argument('--config',type=Path,default=Path('configs/experiments.yaml'))
    parser.add_argument('--profile',choices=['smoke','cpu-light','gpu','full'],default='smoke')
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    parser.add_argument('--seed',type=int,default=None,help='Explicitly restrict to one seed; default uses configured seeds')
    parser.add_argument('--model',choices=['naive','seasonal_daily','seasonal_weekly','arima','xgboost','lstm','fedavg','fedprox'],default='lstm')
    parser.add_argument('--experiment',choices=['1','2','3','4','5','6','all'],default='1')
    parser.add_argument('--backend',choices=['ray','inprocess'],default=None)
    parser.add_argument('--regions',nargs='+')
    parser.add_argument('--partition',choices=['iid-like','temporal','regional'])
    parser.add_argument('--normalization',choices=['common','local'])
    parser.add_argument('--mu',type=float)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--input',type=Path,help='CSV input for inference or prediction evaluation')
    parser.add_argument('--region',default='PJME')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.command=='summarize':summarize(args.output_dir);return
    if args.command=='infer':
        from .inference import infer
        if args.checkpoint is None or args.input is None:parser.error('infer requires --checkpoint and --input')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        infer(args.checkpoint,args.input,args.region).to_csv(args.output_dir/'inference.csv',index=False);return
    if args.command=='evaluate':
        import pandas as pd
        from .evaluation import metrics
        if args.input is None:parser.error('evaluate requires --input predictions.csv')
        f=pd.read_csv(args.input)
        json_write(args.output_dir/'evaluated_metrics.json',metrics(f.actual_mw.to_numpy(),f.prediction_mw.to_numpy()));return
    cfg=read_config(args.config,args.profile)
    if args.backend:cfg['federated']['backend']=args.backend
    if args.regions:cfg['data']['regions']=args.regions
    if args.partition:cfg['federated']['partition']=args.partition
    if args.normalization:cfg['federated']['normalization']=args.normalization
    if args.mu is not None:cfg['federated']['mu']=args.mu
    if args.seed is not None:cfg['seeds']=[args.seed]
    if args.command=='prepare':
        out=run_directory(args.output_dir,'prepare',cfg);prepare(cfg,args.data_dir,out);print(out);return
    if args.command=='experiment':
        nums=range(1,7) if args.experiment=='all' else [int(args.experiment)]
        for number in nums:
            for name,c,shap in experiment_jobs(number,cfg):
                c['experiment']=number
                for seed in c['seeds']:run_one(c,args.data_dir,args.output_dir,name,seed,shap)
            if number==6:summarize(args.output_dir)
    else:
        name='xgboost' if args.command=='shap' else args.model
        if args.command=='federated' and name not in {'fedavg','fedprox'}:
            parser.error('federated requires --model fedavg or fedprox')
        for seed in cfg['seeds']:run_one(cfg,args.data_dir,args.output_dir,name,seed,args.command=='shap')


if __name__=='__main__':main()
