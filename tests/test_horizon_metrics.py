import numpy as np

from compare_horizons import metrics


def test_point_and_prefix_errors_have_distinct_meanings():
    pred = np.arange(1., 11.).reshape(1, 10, 1)
    measured = metrics(pred, np.zeros_like(pred))
    assert measured['point']['2']['mse'] == 4.
    assert measured['prefix']['2']['mse'] == 2.5
    assert measured['point']['10']['mse'] == 100.
    assert measured['prefix']['10']['mse'] == 38.5
    assert measured['point']['10']['mae'] == 10.
    assert measured['prefix']['10']['mae'] == 5.5
