"""Small synthetic integration fixtures: never research-performance evidence."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from pulse_predict.utils import read_config
from pulse_predict.data import make_samples,Scale,training_scales
from pulse_predict.models import fit_arima,fit_lstm,save_lstm,fit_xgboost
from pulse_predict.inference import infer


@pytest.fixture
def synthetic(tmp_path):
    cfg=read_config(Path('configs/experiments.yaml'),'smoke')
    cfg['data'].update(start='2016-01-01',validation_start='2016-01-15',test_start='2016-01-20',end='2016-01-25')
    cfg['lstm'].update(hidden_size=4,epochs=1)
    cfg['xgboost'].update(n_estimators=5)
    cfg['arima'].update(orders=[[1,0,0]])
    idx=pd.date_range('2016-01-01',periods=24*25,freq='h')
    y=100+10*np.sin(np.arange(len(idx))*2*np.pi/24)+np.random.default_rng(4).normal(size=len(idx))
    s=pd.Series(y,index=idx)
    p=tmp_path/'PJME_hourly.csv';pd.DataFrame({'Datetime':idx,'PJME_MW':y}).to_csv(p,index=False)
    return s,cfg,p


@pytest.mark.integration
def test_neural_and_tree_unseen_inference(synthetic,tmp_path):
    s,cfg,p=synthetic;a=make_samples(s,cfg);scale=training_scales({'PJME':a},'common')['PJME']
    model,hist,stats=fit_lstm(scale.transform(a.windows('train')),scale.transform(a.targets('train')),
                              scale.transform(a.windows('validation')),scale.transform(a.targets('validation')),cfg['lstm'],42,'cpu',
                              validation_std=np.full(len(a.targets('validation')),scale.std))
    assert stats['selection_metric']=='rmse_mw'
    assert hist[0]['validation_selection']==pytest.approx(np.sqrt(hist[0]['validation_loss'])*scale.std,rel=1e-6)
    save_lstm(tmp_path/'model.pt',model,{'PJME':scale},24)
    result=infer(tmp_path/'model.pt',p,'PJME')
    assert result.timestamp.max()==s.index.max()+pd.Timedelta(hours=1)
    assert np.isfinite(result.prediction_mw).all()
    tree,stats=fit_xgboost(a,cfg['xgboost'],42,'cpu')
    assert len(tree.predict(a.tabular('test')))==len(a.targets('test'))


@pytest.mark.integration
def test_arima_one_step_future_invariance(synthetic):
    s,cfg,p=synthetic;a=make_samples(s,cfg)
    pred,stats,_=fit_arima(a,cfg['arima'])
    modified=s.copy();first=a.indices['test'][0];modified.iloc[first:]+=999
    b=make_samples(modified,cfg);other,_,_=fit_arima(b,cfg['arima'])
    assert pred[0]==pytest.approx(other[0])


@pytest.mark.integration
def test_flower_message_runtime_mu_zero(synthetic,tmp_path):
    from pulse_predict.federated import fit_federated
    from pulse_predict.models import predict_lstm
    s,cfg,p=synthetic;a=make_samples(s,cfg);samples={'PJME':a};scales=training_scales(samples,'common')
    cfg['federated'].update(backend='inprocess',rounds=2,clients=2,method='fedavg')
    out1=tmp_path/'avg';out2=tmp_path/'prox';out1.mkdir();out2.mkdir()
    avg,stats=fit_federated(samples,scales,cfg,42,'cpu',out1)
    cfg=copy.deepcopy(cfg);cfg['federated'].update(method='fedprox',mu=0.)
    prox,_=fit_federated(samples,scales,cfg,42,'cpu',out2)
    x=scales['PJME'].transform(a.windows('test'))
    np.testing.assert_array_equal(predict_lstm(avg,x,'cpu'),predict_lstm(prox,x,'cpu'))
    assert stats['communication_rounds']==2 and stats['updates']>0


@pytest.mark.ray
def test_ray_runtime(synthetic,tmp_path):
    import os,sys
    if sys.platform=='win32' or os.environ.get('PULSE_TEST_RAY')!='1':
        pytest.skip('Explicit Linux Ray validation: set PULSE_TEST_RAY=1')
    from pulse_predict.federated import fit_federated
    s,cfg,p=synthetic;cfg['federated'].update(backend='ray',rounds=1,clients=2)
    a=make_samples(s,cfg);samples={'PJME':a};scales=training_scales(samples,'common')
    model,stats=fit_federated(samples,scales,cfg,42,'cpu',tmp_path)
    assert stats['communication_rounds']==1
