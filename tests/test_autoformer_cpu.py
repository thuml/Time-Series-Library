"""Regression for upstream's hard-coded CUDA indices during evaluation."""
from types import SimpleNamespace

import torch

from models.Autoformer import Model


def test_original_autoformer_can_evaluate_on_cpu():
    cfg = SimpleNamespace(seq_len=32, pred_len=8, label_len=16, output_attention=False,
                          moving_avg=5, enc_in=2, dec_in=2, c_out=2, d_model=8,
                          embed='timeF', embed_type=0, freq='h', dropout=0.,
                          factor=1, n_heads=2, d_ff=16, e_layers=1, d_layers=1,
                          activation='gelu')
    model = Model(cfg).cpu().eval()
    x = torch.randn(1, 32, 2)
    decoder = torch.cat([x[:, -16:], torch.zeros(1, 8, 2)], 1)
    with torch.no_grad():
        prediction = model(x, torch.zeros(1, 32, 4), decoder, torch.zeros(1, 24, 4))
    assert prediction.shape == (1, 8, 2)
    assert prediction.device.type == 'cpu' and torch.isfinite(prediction).all()
