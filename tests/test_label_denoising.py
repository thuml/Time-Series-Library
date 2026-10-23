import numpy as np
import pytest
from PyEMD import CEEMDAN, EMD

from label_denoise_experiment import Windows, denoise_channel, error_metrics


def test_denoised_targets_retain_all_but_selected_imfs():
    t = np.arange(128.)
    x = np.sin(2*np.pi*t/5) + .4*np.sin(2*np.pi*t/19) + .01*t
    cfg = dict(ceemdan_trials=2, ceemdan_epsilon=.005, ceemdan_seed=421,
               ceemdan_max_sift=10)
    _, clean1, clean2, first, second, _ = denoise_channel((0, x, cfg))
    backend = CEEMDAN(trials=2, epsilon=.005, ext_EMD=EMD(MAX_ITERATION=10), parallel=False)
    backend.noise_seed(421)
    modes = backend.ceemdan(x, max_imf=2)
    np.testing.assert_allclose(clean1, modes[1:].sum(0), atol=1e-12)
    np.testing.assert_allclose(clean2, modes[2:].sum(0), atol=1e-12)
    np.testing.assert_allclose(clean2+first+second, x, atol=1e-12)


def test_raw_input_and_raw_evaluation_target_are_not_denoised():
    raw = np.arange(40., dtype=np.float32).reshape(20, 2)
    label = raw[:12] * .5
    data = Windows(raw, label, [4, 8], 4, 4)
    x, y, raw_y = data[1]
    np.testing.assert_array_equal(x, raw[4:8])
    np.testing.assert_array_equal(y, label[8:12])
    np.testing.assert_array_equal(raw_y, raw[8:12])
    with pytest.raises(ValueError, match='boundary'):
        Windows(raw, label, [10], 4, 4)


def test_evaluation_uses_original_targets_and_original_units_per_channel():
    pred = np.ones((2, 96, 2), dtype=np.float32)
    raw = np.zeros_like(pred)
    value = error_metrics(pred, raw, np.array([2., 3.]))
    assert value['mse'] == 1 and value['mae'] == 1
    assert value['original_units_per_channel_rmse'] == [2., 3.]
    assert value['original_units_per_channel_mae'] == [2., 3.]


def test_constant_training_signal_is_preserved():
    cfg = dict(ceemdan_trials=2, ceemdan_epsilon=.005, ceemdan_seed=421,
               ceemdan_max_sift=10)
    x = np.ones(64)
    _, a, b, first, second, _ = denoise_channel((0, x, cfg))
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, x)
    assert not first.any() and not second.any()
