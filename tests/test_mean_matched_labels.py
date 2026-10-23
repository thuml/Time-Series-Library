import numpy as np
import pytest

from mean_matched_label_experiment import mean_match_coefficients, window_means


def test_round_coefficients_before_applying_to_original_units():
    raw=np.array([[2.,3.],[4.,7.],[6.,11.]])
    clean=raw*np.array([1.012345, .987654])
    coefficients, corrected=mean_match_coefficients(raw,clean)
    expected=np.array([float(f'{x:.4f}') for x in raw.mean(0)/clean.mean(0)])
    np.testing.assert_array_equal(coefficients,expected)
    np.testing.assert_array_equal(corrected,clean*expected)
    assert np.max(abs(corrected.mean(0)-raw.mean(0))/raw.mean(0))<.0001


def test_impossible_multiplicative_mean_match_is_rejected():
    with pytest.raises(ValueError,match='zero-mean'):
        mean_match_coefficients(np.ones((2,1)),np.array([[-1.],[1.]]))


def test_window_mean_matches_explicit_target_window_average():
    values=np.arange(20.).reshape(10,2)
    origins=np.array([2,3,6])
    expected=np.stack([values[t:t+3] for t in origins]).mean((0,1))
    np.testing.assert_allclose(window_means(values,origins,3),expected)
