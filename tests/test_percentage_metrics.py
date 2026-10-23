import numpy as np
import pytest

from reevaluate_denoised_targets import percentage_metrics


def test_mape_excludes_zero_but_wmape_counts_its_error():
    target = np.array([0.,1.,2.]).reshape(1,3,1)
    pred = np.array([1.,2.,1.]).reshape(1,3,1)
    r = percentage_metrics(pred,target,np.array([1e-8]))
    assert r['mape_macro_percent']==75.
    assert r['wmape_macro_percent']==100.
    assert r['mape_excluded_per_channel']==[1]


def test_macro_is_invariant_to_per_channel_unit_changes():
    y = np.array([1.,2.,3.,5.]).reshape(1,2,2)
    p = y+np.array([.5,2.])[None,None,:]
    r = percentage_metrics(p,y,0.)
    units = np.array([100.,1.])[None,None,:]
    s = percentage_metrics(p*units,y*units,0.)
    assert s['mape_macro_percent']==pytest.approx(r['mape_macro_percent'])
    assert s['wmape_macro_percent']==pytest.approx(r['wmape_macro_percent'])
    assert s['wmape_pooled_percent']!=pytest.approx(r['wmape_pooled_percent'])


def test_negative_targets_use_absolute_denominators():
    y=np.array([-2.,-4.]).reshape(1,2,1)
    p=np.array([-1.,-2.]).reshape(1,2,1)
    r=percentage_metrics(p,y,0.)
    assert r['mape_macro_percent']==50.
    assert r['wmape_macro_percent']==50.
