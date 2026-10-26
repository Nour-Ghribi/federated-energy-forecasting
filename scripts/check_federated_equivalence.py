"""Check exact mu=0/FedAvg agreement in completed functional artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('results/validated'))
    args=parser.parse_args()
    avg={};prox={}
    for path in sorted(args.results.glob('*/config.json')):
        cfg=json.loads(path.read_text(encoding='utf-8'))
        if cfg.get('model') not in {'fedavg','fedprox'}:continue
        if cfg['federated']['backend']!='inprocess':continue
        status=path.parent/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='completed':continue
        fed={k:v for k,v in cfg['federated'].items() if k not in {'method','mu'}}
        version=json.loads((path.parent/'environment.json').read_text())['package_source_sha256']
        key=json.dumps([cfg['data'],cfg['lstm'],cfg['seed'],cfg['device'],fed,version],sort_keys=True)
        if cfg['model']=='fedavg':avg[key]=path.parent
        elif cfg['federated']['mu']==0:prox[key]=path.parent
    if not prox:raise RuntimeError('No completed mu=0 artifacts')
    results=[]
    for key,p in prox.items():
        if key not in avg:raise RuntimeError(f'Missing equivalent FedAvg run for {p}')
        for f in p.glob('predictions_*.csv'):
            g=avg[key]/f.name;a=pd.read_csv(f);b=pd.read_csv(g)
            if not a.timestamp.equals(b.timestamp) or not np.array_equal(a.actual_mw,b.actual_mw):
                raise RuntimeError('Evaluation observations/timestamps differ')
            exact=np.array_equal(a.prediction_mw,b.prediction_mw)
            result={'fedprox':str(f.relative_to(args.results)),'fedavg':str(g.relative_to(args.results)),
                    'samples':len(a),'exact_predictions_equal':exact,
                    'maximum_absolute_difference_mw':float(np.abs(a.prediction_mw-b.prediction_mw).max())}
            results.append(result)
            if not exact:raise RuntimeError(f'mu=0 disagreement: {result}')
    report={'purpose':'functional_validation_only','protocol_pairs':len(prox),
            'regional_comparisons':len(results),'all_exact':True,'comparisons':results}
    out=args.results/'fedprox_zero_real_checks.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{len(prox)} protocol pairs, {len(results)} regional comparisons: exact agreement. {out}')


if __name__=='__main__':main()
