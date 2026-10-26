"""Run and record real validation; smoke scores are never research findings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--matrix',action='store_true',help='Run all six experiments at three seeds in functional smoke mode')
    parser.add_argument('--output-dir',type=Path,default=Path('results/validated'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    def fingerprints():
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csv')}
    before=fingerprints();report={'purpose':'functional_validation_only','commands':[],'raw_sha256_before':before}
    commands=[[sys.executable,'-m','pip','check'],
              [sys.executable,'-m','pytest','-q','--junitxml',str(output/'pytest.xml')],
              [sys.executable,'-m','pulse_predict.cli','prepare','--profile','smoke','--output-dir',str(output)]]
    base=[sys.executable,'-m','pulse_predict.cli']
    shared=['--profile','smoke','--backend','inprocess','--output-dir',str(output)]
    if args.matrix:
        commands.append(base+['experiment','--experiment','all']+shared)
    else:
        commands.extend([base+['experiment','--experiment','1','--seed','42']+shared,
                         base+['experiment','--experiment','3','--seed','42']+shared,
                         base+['shap','--seed','42']+shared,
                         base+['summarize','--output-dir',str(output)]])
    env=os.environ.copy();env['FLWR_TELEMETRY_ENABLED']='0';env['PYTHONIOENCODING']='utf-8'
    for i,command in enumerate(commands):
        print(f'Validation step {i+1}/{len(commands)}: '+ ' '.join(command[1:]),flush=True)
        started=time.perf_counter();log=output/f'command_{i+1}.log'
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,env=env)
        report['commands'].append({'argv':command,'returncode':result.returncode,'seconds':time.perf_counter()-started,'log':log.name})
        (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-6000:]);raise SystemExit(result.returncode)
    import pandas as pd
    import nbformat
    from pulse_predict.evaluation import assert_manifest
    runs=[]
    for metrics in output.glob('*/metrics.csv'):
        path=metrics.parent
        for pred in path.glob('predictions_*.csv'):
            region=pred.stem.removeprefix('predictions_')
            manifest=pd.read_csv(path/f'manifest_{region}.csv')
            expected=pd.DatetimeIndex(pd.to_datetime(manifest.loc[manifest.split=='test','timestamp']))
            assert_manifest(expected,pd.read_csv(pred))
        runs.append(path.name)
    for p in (root/'notebooks').glob('*.ipynb'):nbformat.validate(nbformat.read(p,as_version=4))
    after=fingerprints();report.update(raw_sha256_after=after,raw_files_unchanged=before==after,
                                      completed_runs=len(runs),verified_prediction_manifests=True,notebook_schemas_valid=True)
    if before!=after:raise RuntimeError('Original CSV files changed during validation')
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Validated {len(runs)} completed runs; original CSV hashes unchanged. Report: {output / "validation.json"}',flush=True)


if __name__=='__main__':main()
