"""Flower Message API applications, explicit Ray or in-process functional runtime."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from .data import Samples, Scale
from .models import seed_everything, new_lstm, train_epoch, predict_lstm
from .evaluation import metrics


@dataclass
class ClientData:
    region: str
    x: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex


def partition_indices(n: int, count: int, kind: str, seed: int) -> list[np.ndarray]:
    if n < count or count < 1:
        raise ValueError("Each client needs at least one sample")
    if kind == "iid-like":
        order=np.random.default_rng(seed).permutation(n)
    elif kind == "temporal":
        order=np.arange(n)
    else:
        raise ValueError(f"Unknown partition {kind}")
    return [np.sort(v) for v in np.array_split(order,count)]


def clients_from_samples(samples: dict[str, Samples], scales: dict[str, Scale], cfg: dict) -> list[ClientData]:
    kind=cfg['federated']['partition']; result=[]
    for region,s in samples.items():
        x=scales[region].transform(s.windows('train')); y=scales[region].transform(s.targets('train'))
        if kind=='regional':
            parts=[np.arange(len(y))]
        else:
            if len(samples)!=1: raise ValueError('Single-region partitions require exactly one region')
            parts=partition_indices(len(y),cfg['federated']['clients'],kind,cfg['federated']['partition_seed'])
        for pos in parts:
            result.append(ClientData(region,x[pos],y[pos],s.timestamps('train')[pos]))
    if len(result)!=cfg['federated']['clients']:
        raise ValueError('Client count must match number of regional datasets')
    return result


def diagnostics(clients: list[ClientData], scales: dict[str, Scale]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows=[]; hourly=[]; values=[]
    for i,c in enumerate(clients):
        y=scales[c.region].inverse(c.y);values.append(y)
        rows.append({'client':i,'region':c.region,'samples':len(y),'start':str(c.timestamps.min()),
                     'end':str(c.timestamps.max()),'mean_mw':float(y.mean()),'std_mw':float(y.std()),
                     **{f'q{q}_mw':float(np.percentile(y,q)) for q in [5,25,50,75,95]}})
        hourly.extend({'client':i,'hour':int(h),'mean_mw':float(v)} for h,v in pd.Series(y,index=c.timestamps).groupby(c.timestamps.hour).mean().items())
    denom=max(float(np.std(np.concatenate(values))),1e-8)
    distances=[{'client_a':i,'client_b':j,'wasserstein_mw':float(wasserstein_distance(a,b)),
                'wasserstein_normalized':float(wasserstein_distance(a,b)/denom)}
               for i,a in enumerate(values) for j,b in enumerate(values) if i<j]
    return pd.DataFrame(rows),pd.DataFrame(hourly),pd.DataFrame(distances)


def weighted_parameters(states: list[list[np.ndarray]], counts: list[int]) -> list[np.ndarray]:
    """Reference implementation for aggregation regression tests, not a runtime."""
    if not states or len(states)!=len(counts) or min(counts)<=0:
        raise ValueError('Invalid aggregation inputs')
    if any(len(s)!=len(states[0]) for s in states):raise ValueError('Mismatched states')
    output=[]
    for tensors in zip(*states,strict=True):
        if any(a.shape!=tensors[0].shape for a in tensors):raise ValueError('Mismatched tensors')
        output.append(sum(a*(n/sum(counts)) for a,n in zip(tensors,counts,strict=True)))
    return output


def make_client_app(clients: list[ClientData], cfg: dict, seed: int, device: str):
    from flwr.app import ArrayRecord, MetricRecord, RecordDict, Message
    from flwr.clientapp import ClientApp
    app=ClientApp()

    @app.train()
    def train(msg,context):
        cid=int(context.node_config['partition-id'])
        c=clients[cid]; server_round=int(msg.content['config']['server-round'])
        local_seed=seed+server_round*10000+cid*100
        seed_everything(local_seed,device)
        model=new_lstm(cfg['lstm'],device)
        model.load_state_dict(msg.content['arrays'].to_torch_state_dict())
        reference=[p.detach().clone() for p in model.parameters() if p.requires_grad]
        mu=float(msg.content['config'].get('proximal-mu',0.0))
        updates=0;examples=0;loss=0.0;started=time.perf_counter()
        # Adam state persists within a round and resets between rounds.
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lstm']['learning_rate'])
        for epoch in range(cfg['federated']['local_epochs']):
            stats=train_epoch(model,c.x,c.y,cfg['lstm'],device,local_seed+epoch,mu,reference,optimizer)
            updates+=stats['updates'];examples+=stats['examples_processed'];loss=stats['loss']
        content=RecordDict({'arrays':ArrayRecord(model.cpu().state_dict()),
                            'metrics':MetricRecord({'num-examples':len(c.y),'train_loss':loss,
                                                    'updates':updates,'examples_processed':examples,
                                                    'client_seconds':time.perf_counter()-started})})
        return Message(content=content,reply_to=msg)
    return app


class FunctionalGrid:
    """Synchronous test transport for actual Flower apps/strategies.

    Not Ray, not a distributed deployment and never selected as an automatic
    fallback. Only --backend inprocess opts into this functional diagnostic.
    """
    def __init__(self,app,count:int):
        from flwr.app import Context,RecordDict
        self.app=app
        self.contexts={i+1:Context(run_id=1,node_id=i+1,node_config={'partition-id':i,'num-partitions':count},
                                   state=RecordDict(),run_config={}) for i in range(count)}

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self,messages,*,timeout=None):
        return [self.app(msg,self.contexts[msg.metadata.dst_node_id]) for msg in messages]


def fit_federated(samples: dict[str,Samples],scales:dict[str,Scale],cfg:dict,seed:int,device:str,out:Path):
    from flwr.app import ArrayRecord,ConfigRecord,MetricRecord
    from flwr.serverapp.strategy import FedAvg,FedProx
    f=cfg['federated']; backend=f['backend']
    if backend not in {'ray','inprocess'}:raise ValueError('backend must be ray or inprocess')
    if backend=='inprocess' and cfg['purpose']!='functional_validation_only':
        raise ValueError('inprocess transport is restricted to functional validation')
    if backend=='ray':
        import sys,importlib.util
        if sys.platform=='win32':
            raise RuntimeError('Run the Ray research runtime inside Linux/WSL2, not the Windows interpreter. Use explicit --backend inprocess only for smoke validation.')
        if importlib.util.find_spec('ray') is None:
            raise RuntimeError('Install requirements-simulation.txt in the Linux environment')
    clients=clients_from_samples(samples,scales,cfg)
    pd.concat([pd.DataFrame({'client':i,'region':c.region,'timestamp':c.timestamps}) for i,c in enumerate(clients)],ignore_index=True).to_csv(out/'client_training_manifest.csv',index=False)
    tables=diagnostics(clients,scales)
    for name,frame in zip(('clients','hourly_profiles','distribution_distances'),tables,strict=True):
        frame.to_csv(out/f'{name}.csv',index=False)
    from .figures import client_diagnostics
    client_diagnostics(tables[1],tables[2],out/'figures',cfg['purpose'])
    app=make_client_app(clients,cfg,seed,device)
    seed_everything(seed,device);initial=new_lstm(cfg['lstm'],'cpu')
    history=[];best_score=float('inf');best_state=None
    started=time.perf_counter()

    def evaluate(server_round,arrays):
        nonlocal best_score,best_state
        model=new_lstm(cfg['lstm'],device);model.load_state_dict(arrays.to_torch_state_dict())
        errors=[];truth=[];per={}
        for region,s in samples.items():
            pred=scales[region].inverse(predict_lstm(model,scales[region].transform(s.windows('validation')),device))
            y=s.targets('validation');m=metrics(y,pred);per[f'{region}_mae']=m['mae']
            errors.append(pred-y);truth.append(y)
        y=np.concatenate(truth);pred=y+np.concatenate(errors);m=metrics(y,pred)
        history.append({'round':server_round,'validation_mae':m['mae'],'validation_rmse':m['rmse'],
                        'elapsed_seconds':time.perf_counter()-started,**per})
        criterion='rmse' if cfg['lstm'].get('loss','mse')=='mse' else 'mae'
        if server_round>0 and m[criterion]<best_score:
            best_score=m[criterion];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        return MetricRecord(m)

    count=len(clients)
    strategy_cls=FedProx if f['method']=='fedprox' else FedAvg
    args=dict(fraction_train=1.0,fraction_evaluate=0.0,min_train_nodes=count,
              min_evaluate_nodes=0,min_available_nodes=count)
    if f['method']=='fedprox':args['proximal_mu']=f['mu']
    strategy=strategy_cls(**args)

    def execute(grid):
        result=strategy.start(grid=grid,initial_arrays=ArrayRecord(initial.state_dict()),num_rounds=f['rounds'],
                              train_config=ConfigRecord({}),evaluate_fn=evaluate)
        if best_state is None:raise RuntimeError('Federated training produced no valid checkpoint')
        torch.save(best_state,out/'federated_best_state.pt')
        pd.DataFrame(history).to_csv(out/'rounds.csv',index=False)
        # Weighted Flower diagnostics are retained; total counters computed below.
        (out/'flower_result.txt').write_text(str(result),encoding='utf-8')

    if backend=='inprocess':
        # Flower 1.40's Message constructor reads process task identity. The
        # official runtime sets it; this opt-in test transport mirrors that
        # initialization and restores the prior identity afterwards. This is
        # the only adapter coupled to a Flower internal (version pinned).
        from flwr.supercore.task_identity import TaskIdentity
        previous=(TaskIdentity._run_id,TaskIdentity._node_id,TaskIdentity._task_id)
        TaskIdentity.run_id=1;TaskIdentity.node_id=0;TaskIdentity.task_id=1
        try:
            execute(FunctionalGrid(app,count))
        finally:
            TaskIdentity._run_id,TaskIdentity._node_id,TaskIdentity._task_id=previous
    else:
        from flwr.serverapp import ServerApp
        from flwr.simulation import run_simulation
        server=ServerApp()
        @server.main()
        def main(grid,context):execute(grid)
        run_simulation(server_app=server,client_app=app,num_supernodes=count,
                       backend_config={'init_args':{'num_cpus':2,'include_dashboard':False},
                                       'client_resources':{'num_cpus':2,'num_gpus':1.0 if device=='cuda' else 0.0}})
    model=new_lstm(cfg['lstm'],device)
    model.load_state_dict(torch.load(out/'federated_best_state.pt',map_location=device,weights_only=True))
    rounds=pd.read_csv(out/'rounds.csv')
    payload=sum(p.numel()*p.element_size() for p in initial.state_dict().values())
    stats={'training_seconds':time.perf_counter()-started,'communication_rounds':f['rounds'],
           'local_epochs':f['local_epochs'],'train_samples':sum(len(c.y) for c in clients),
           'examples_processed':sum(len(c.y) for c in clients)*f['rounds']*f['local_epochs'],
           'updates':sum(int(np.ceil(len(c.y)/cfg['lstm']['batch_size'])) for c in clients)*f['rounds']*f['local_epochs'],
           'simulated_parameter_payload_bytes':int(2*count*f['rounds']*payload),
           'selected_round':int(rounds.loc[rounds['validation_rmse' if cfg['lstm'].get('loss','mse')=='mse' else 'validation_mae'].where(rounds['round']>0).idxmin(),'round']),
           'selection_metric':('rmse' if cfg['lstm'].get('loss','mse')=='mse' else 'mae')+'_mw',
           'backend':backend,'payload_excludes':'metadata, serialization overhead and evaluation traffic'}
    return model,stats
