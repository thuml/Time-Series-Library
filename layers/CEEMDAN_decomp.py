"""True CEEMDAN in the forward pass; an explicit stop-gradient remainder.

No hidden-state results are cached: they change at every optimizer step.
Only the EMD of identical fixed-seed noise is reused within one backend.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn
from PyEMD import CEEMDAN, EMD


@dataclass(frozen=True)
class CEEMDANConfig:
    trials: int = 8
    epsilon: float = 0.005
    max_imf: int = 3
    max_sift: int = 50
    seed: int = 421

    def __post_init__(self):
        if self.trials < 1 or self.max_imf < 1 or self.max_sift < 1:
            raise ValueError('trials, max_imf and max_sift must be positive')
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('epsilon must be finite and nonnegative')
        if not 0 <= self.seed < 2**32:
            raise ValueError('seed must be in [0, 2**32)')


class _NoiseReuseCEEMDAN(CEEMDAN):
    """An exact noise-only optimization, tested against unmodified PyEMD."""
    def _decompose_noise(self):
        key = (self.all_noises.shape, hashlib.sha256(self.all_noises.tobytes()).digest())
        if getattr(self, '_noise_key', None) != key:
            modes = super()._decompose_noise()
            self._noise_key = key
            self._noise_modes = tuple(m.copy() for m in modes)
        # PyEMD clears the returned list at the end of a decomposition.
        return [m.copy() for m in self._noise_modes]


class CEEMDANDecomposition(nn.Module):
    """[instances, time, features] -> oscillation sum, final remainder.

    Each instance/feature is decomposed independently. The same seed is reset
    for each series, so batch size, ordering and channel ordering cannot alter
    its noise realization. Constant and monotone inputs have no oscillatory
    IMF and are returned entirely as remainder.

    Backward convention: r = stop_gradient(CEEMDAN_remainder(x)); s = x-r.
    Therefore ds/dx=I and dr/dx=0. This is a surrogate, not the derivative of
    CEEMDAN. Prediction heads on r remain trainable.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CEEMDANConfig()
        self._backend = None
        self.last_stats = {}

    def _get_backend(self):
        if self._backend is None:
            c = self.config
            self._backend = _NoiseReuseCEEMDAN(
                trials=c.trials, epsilon=c.epsilon,
                ext_EMD=EMD(MAX_ITERATION=c.max_sift), parallel=False)
        return self._backend

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 3 or not x.is_floating_point():
            raise ValueError('Expected floating [instances, time>=3, features]')
        signals = x.detach().to(device='cpu', dtype=torch.float64).numpy()
        if not np.isfinite(signals).all():
            raise ValueError('Nonfinite CEEMDAN input')
        remainder = np.empty_like(signals)
        calls = 0
        for b in range(signals.shape[0]):
            for d in range(signals.shape[2]):
                signal = signals[b, :, d]
                delta = np.diff(signal)
                if np.std(signal) < 1e-10 or np.all(delta >= 0) or np.all(delta <= 0):
                    remainder[b, :, d] = signal
                    continue
                ce = self._get_backend()
                ce.noise_seed(self.config.seed)
                modes = ce.ceemdan(signal, max_imf=self.config.max_imf)
                # Last row is the remainder even if max_imf truncates extraction.
                remainder[b, :, d] = signal - modes[:-1].sum(axis=0)
                calls += 1
        if not np.isfinite(remainder).all():
            raise RuntimeError('CEEMDAN produced a nonfinite remainder')
        trend = torch.as_tensor(remainder, device=x.device, dtype=x.dtype)
        self.last_stats = {'series': signals.shape[0] * signals.shape[2],
                           'ceemdan_calls': calls, 'length': signals.shape[1]}
        return x - trend, trend


class MovingAverageDecomposition(nn.Module):
    """Autoformer-style differentiable decomposition for an ablation."""
    def __init__(self, kernel_size=5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('moving-average kernel must be positive and odd')
        self.kernel_size = kernel_size

    def forward(self, x):
        z = torch.nn.functional.pad(x.transpose(1, 2),
                                   (self.kernel_size // 2,) * 2, mode='replicate')
        trend = torch.nn.functional.avg_pool1d(z, self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend


class IdentityDecomposition(nn.Module):
    def forward(self, x):
        return x, torch.zeros_like(x)


def make_decomposition(kind, config, kernel_size):
    if kind == 'ceemdan':
        return CEEMDANDecomposition(config)
    if kind == 'moving_avg':
        return MovingAverageDecomposition(kernel_size)
    if kind == 'none':
        return IdentityDecomposition()
    raise ValueError(f'Unknown decomposition: {kind}')
