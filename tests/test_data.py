from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from pulse_predict.data import features, make_samples, training_scales, load_series, align_regions, SlidingWindowDataset
from pulse_predict.evaluation import metrics, assert_manifest, regional_aggregate


@pytest.fixture
def fixture_data():
    idx = pd.date_range('2016-01-01', periods=24*100, freq='h')
    s = pd.Series(100+np.sin(np.arange(len(idx))/24), index=idx)
    cfg = {'data': {'start':'2016-01-01','validation_start':'2016-02-01','test_start':'2016-03-01',
                    'end':'2016-04-01','lags':[1,24,48,168],'rolling':[24,168],'comparison_history':168},
           'lstm': {'sequence_length':24}}
    return s, cfg


def test_future_feature_invariance(fixture_data):
    s,cfg = fixture_data
    x = features(s,cfg['data']['lags'],cfg['data']['rolling'])
    changed=s.copy(); changed.iloc[700:]=99999
    z=features(changed,cfg['data']['lags'],cfg['data']['rolling'])
    pd.testing.assert_frame_equal(x.iloc[:701],z.iloc[:701])
    assert x.lag_1.iloc[200]==s.iloc[199]
    assert x.mean_24.iloc[200]==pytest.approx(s.iloc[176:200].mean())


def test_chronology_windows_and_scaling(fixture_data):
    s,cfg=fixture_data; a=make_samples(s,cfg)
    assert a.timestamps('train').max()<a.timestamps('validation').min()<a.timestamps('test').min()
    w=a.windows('train'); i=a.indices['train'][0]
    assert w.shape==(len(a.indices['train']),24,1)
    np.testing.assert_allclose(w[0,:,0],s.iloc[i-24:i],rtol=1e-6)
    changed=s.copy(); changed.loc['2016-02-01':]=99999
    b=make_samples(changed,cfg)
    for mode in ('common','local'):
        assert training_scales({'A':a,'B':a},mode)==training_scales({'A':b,'B':b},mode)
    # Validation is allowed to use training observations, never its target/future.
    j=a.indices['validation'][0]
    np.testing.assert_allclose(a.windows('validation')[0,:,0],s.iloc[j-24:j],rtol=1e-6)


def test_missing_and_duplicates(tmp_path):
    p=tmp_path/'sample.csv'
    p.write_text('Datetime,X_MW\n2016-01-01 00:00:00,2\n2016-01-01 01:00:00,3\n2016-01-01 01:00:00,4\n2016-01-01 03:00:00,5\n')
    s,a=load_series(p)
    assert s.isna().sum()==2 and a['duplicate_excess']==1
    assert s.iloc[0]==2 and s.iloc[-1]==5


def test_gap_excludes_windows(fixture_data):
    s,cfg=fixture_data; s.iloc[400]=np.nan
    a=make_samples(s,cfg)
    assert not any(400 <= i <= 568 for i in a.indices['train'])


def test_metrics_manifest():
    m=metrics(np.array([0,2]),np.array([0,4]))
    assert m['mae']==1 and m['rmse']==pytest.approx(np.sqrt(2))
    assert m['smape']==pytest.approx(100/3)
    idx=pd.date_range('2016-01-01',periods=2,freq='h')
    f=pd.DataFrame({'timestamp':idx,'actual_mw':[0,2],'prediction_mw':[0,4]})
    assert_manifest(idx,f)
    with pytest.raises(ValueError): assert_manifest(idx,f.iloc[:1])
    with pytest.raises(ValueError): metrics(np.array([1]),np.array([np.nan]))


def test_same_manifest_for_sequence_sensitivity(fixture_data):
    s,cfg=fixture_data
    short=make_samples(s,cfg,24);long=make_samples(s,cfg,168)
    for split in short.indices:
        np.testing.assert_array_equal(short.indices[split],long.indices[split])
    modified=s.copy();cut=short.indices['validation'][0];modified.iloc[cut:]=99999
    other=make_samples(modified,cfg,24)
    np.testing.assert_array_equal(short.windows('validation')[0],other.windows('validation')[0])


def test_regional_common_scaler_not_local(fixture_data):
    s,cfg=fixture_data;a=make_samples(s,cfg);b=make_samples(s*10,cfg)
    common=training_scales({'A':a,'B':b},'common');local=training_scales({'A':a,'B':b},'local')
    assert common['A']==common['B'] and local['A']!=local['B']
    for mode in ['common','local']:
        scales=training_scales({'A':a,'B':b},mode)
        for region,sm in {'A':a,'B':b}.items():
            np.testing.assert_allclose(scales[region].inverse(scales[region].transform(sm.targets('test'))),sm.targets('test'),rtol=1e-6)


def test_regional_alignment_and_rmse(fixture_data):
    s,cfg=fixture_data;a=make_samples(s,cfg);t=s.copy();t.iloc[1700]=np.nan;b=make_samples(t,cfg)
    align_regions({'A':a,'B':b})
    for split in a.indices:assert a.timestamps(split).equals(b.timestamps(split))
    a=pd.DataFrame({'actual_mw':[0.],'prediction_mw':[2.]})
    b=pd.DataFrame({'actual_mw':[0.,0.,0.],'prediction_mw':[4.,4.,4.]})
    result=regional_aggregate([a,b])
    assert result['macro']['rmse']==3
    assert result['pooled']['rmse']==pytest.approx(np.sqrt(13))


def test_schema_requires_explicit_multiregion_target(tmp_path):
    p=tmp_path/'multi.csv';p.write_text('Datetime,A,B\n2016-01-01 00:00:00,2,3\n')
    with pytest.raises(ValueError,match='target'):load_series(p)
    s,_=load_series(p,target='A');assert s.iloc[0]==2
    with pytest.raises(ValueError,match='MW'):load_series(p,target='A',units='kWh')


def test_dataset_window_boundaries():
    d=SlidingWindowDataset(np.arange(20),np.array([5,10]),5)
    np.testing.assert_array_equal(d[0][0][:,0],np.arange(5))
    assert d[1][1]==10 and len(d)==2
    with pytest.raises(ValueError):SlidingWindowDataset(np.arange(10),np.array([2]),5)
