import copy
import numpy as np
import torch
import pytest
from pulse_predict.models import LSTMForecaster,proximal_penalty,seed_everything,train_epoch,save_lstm,load_lstm,predict_lstm,resolve_device
from pulse_predict.data import Scale
from pulse_predict.federated import weighted_parameters,partition_indices


def test_lstm_checkpoint(tmp_path):
    seed_everything(42);model=LSTMForecaster(hidden_size=4)
    x=np.ones((7,24,1),dtype=np.float32)
    assert model(torch.from_numpy(x)).shape==(7,)
    before=predict_lstm(model,x,'cpu')
    save_lstm(tmp_path/'model.pt',model,{'PJME':Scale(100,5)},24)
    loaded,scales,length=load_lstm(tmp_path/'model.pt')
    np.testing.assert_array_equal(before,predict_lstm(loaded,x,'cpu'))
    assert length==24 and scales['PJME'].inverse(np.array([2]))[0]==110


def test_proximal_gradient_and_zero_equivalence():
    seed_everything(42);a=LSTMForecaster(hidden_size=4);b=copy.deepcopy(a)
    reference=[p.detach().clone() for p in a.parameters()]
    with torch.no_grad():
        for p in a.parameters():p.add_(.1)
    loss=proximal_penalty(a,reference,.2);loss.backward()
    for p,r in zip(a.parameters(),reference,strict=True):
        torch.testing.assert_close(p.grad,.2*(p-r))
        assert r.grad is None
    x=np.random.default_rng(2).normal(size=(12,4,1)).astype(np.float32)
    y=np.arange(12,dtype=np.float32)
    a=copy.deepcopy(b);cfg={'batch_size':4,'learning_rate':.01,'gradient_clip':1.0}
    seed_everything(9);train_epoch(a,x,y,cfg,'cpu',123)
    seed_everything(9);train_epoch(b,x,y,cfg,'cpu',123,mu=0.,reference=reference)
    for p,q in zip(a.parameters(),b.parameters(),strict=True):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_aggregation_and_partitions():
    result=weighted_parameters([[np.array([1.,2.])],[np.array([5.,6.])]], [1,3])
    np.testing.assert_allclose(result[0],[4,5])
    for kind in ['iid-like','temporal']:
        a=partition_indices(101,4,kind,42);b=partition_indices(101,4,kind,42)
        np.testing.assert_array_equal(np.sort(np.concatenate(a)),np.arange(101))
        for x,y in zip(a,b,strict=True):
            np.testing.assert_array_equal(x,y);assert (np.diff(x)>0).all()
    with pytest.raises(ValueError):partition_indices(2,4,'temporal',1)


def test_gpu_explicit():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='CUDA'):resolve_device('cuda')
