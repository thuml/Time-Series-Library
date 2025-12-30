import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# ============ Probabilistic Prediction Metrics ============

def CRPS(samples, true):
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS measures the quality of probabilistic predictions by comparing
    the predicted CDF to the empirical CDF of the true values.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        true: [B, pred_len, N] ground truth
    Returns:
        crps: scalar CRPS value (lower is better)
    """
    n_samples = samples.shape[0]

    # Sort samples along sample dimension
    samples_sorted = np.sort(samples, axis=0)

    # Compute CRPS using energy score approximation
    # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    # where X, X' are independent samples from the forecast distribution

    # First term: mean absolute error between samples and true
    mae_term = np.mean(np.abs(samples - true[np.newaxis, ...]))

    # Second term: mean pairwise distance between samples
    pairwise_sum = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairwise_sum += np.mean(np.abs(samples[i] - samples[j]))
    pairwise_term = 2.0 * pairwise_sum / (n_samples * (n_samples - 1)) if n_samples > 1 else 0.0

    crps = mae_term - 0.5 * pairwise_term
    return crps


def CRPS_empirical(samples, true):
    """
    Compute CRPS using empirical CDF method.

    More efficient implementation for large sample sizes.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        true: [B, pred_len, N] ground truth
    Returns:
        crps: scalar CRPS value
    """
    n_samples = samples.shape[0]

    # Sort samples
    samples_sorted = np.sort(samples, axis=0)

    # Compute CRPS as sum over all samples
    crps = 0.0
    for i in range(n_samples):
        # Weight for this sample in the CDF
        weight = (2 * (i + 1) - 1 - n_samples) / n_samples
        crps += weight * (samples_sorted[i] - true)

    crps = np.mean(np.abs(crps)) / n_samples
    return crps


def calibration(samples, true, coverage_levels=[0.5, 0.9]):
    """
    Compute calibration (empirical coverage of prediction intervals).

    A well-calibrated model should have empirical coverage close to nominal coverage.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        true: [B, pred_len, N] ground truth
        coverage_levels: list of nominal coverage levels (e.g., [0.5, 0.9])
    Returns:
        dict of coverage level -> empirical coverage
    """
    n_samples = samples.shape[0]
    results = {}

    samples_sorted = np.sort(samples, axis=0)

    for level in coverage_levels:
        alpha = 1 - level
        lower_idx = int(n_samples * alpha / 2)
        upper_idx = int(n_samples * (1 - alpha / 2)) - 1

        lower_idx = max(0, lower_idx)
        upper_idx = min(n_samples - 1, upper_idx)

        lower = samples_sorted[lower_idx]
        upper = samples_sorted[upper_idx]

        # Compute empirical coverage
        within = ((true >= lower) & (true <= upper)).astype(float).mean()
        results[f'coverage_{int(level*100)}'] = within

    return results


def sharpness(samples):
    """
    Compute sharpness (average width of prediction intervals).

    Lower sharpness indicates more confident predictions.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
    Returns:
        mean_std: average standard deviation across all predictions
    """
    return np.std(samples, axis=0).mean()


def quantile_loss(samples, true, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute quantile loss (pinball loss) at specified quantiles.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        true: [B, pred_len, N] ground truth
        quantiles: list of quantiles to evaluate
    Returns:
        dict of quantile -> loss value
    """
    n_samples = samples.shape[0]
    samples_sorted = np.sort(samples, axis=0)
    results = {}

    for q in quantiles:
        idx = int(n_samples * q)
        idx = min(max(0, idx), n_samples - 1)
        pred_q = samples_sorted[idx]

        # Pinball loss
        error = true - pred_q
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        results[f'ql_{int(q*100)}'] = np.mean(loss)

    return results


def prob_metric(samples, true):
    """
    Compute all probabilistic metrics.

    Args:
        samples: [n_samples, B, pred_len, N] sampled predictions
        true: [B, pred_len, N] ground truth
    Returns:
        dict with all probabilistic metrics
    """
    crps = CRPS(samples, true)
    calib = calibration(samples, true)
    sharp = sharpness(samples)
    ql = quantile_loss(samples, true)

    # Mean prediction for point metrics
    mean_pred = np.mean(samples, axis=0)
    mae = MAE(mean_pred, true)
    mse = MSE(mean_pred, true)
    rmse = RMSE(mean_pred, true)

    results = {
        'crps': crps,
        'sharpness': sharp,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        **calib,
        **ql
    }
    return results
