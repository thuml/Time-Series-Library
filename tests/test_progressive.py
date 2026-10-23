from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PyEMD import CEEMDAN, EMD

from layers.CEEMDAN_decomp import CEEMDANConfig, CEEMDANDecomposition
from layers.CEEMDAN_PatchTST_backbone import InstanceAutoCorrelation
from models.CEEMDANPatchTST import Model
from models.PatchTST import Model as OfficialPatchTST


def config(**overrides):
    values = dict(enc_in=2, seq_len=32, pred_len=8, e_layers=2, n_heads=2,
                  d_model=4, d_ff=8, dropout=0., fc_dropout=0., head_dropout=0.,
                  individual=False, patch_len=8, stride=4, padding_patch='end',
                  revin=True, affine=True, subtract_last=False, decomposition=False,
                  kernel_size=25, factor=1, encoder_norm='LayerNorm',
                  ceemdan_trials=2, ceemdan_max_imf=1, ceemdan_max_sift=10,
                  fusion_attention='hybrid', progressive_decomp='ceemdan')
    values.update(overrides)
    return SimpleNamespace(**values)


def signal(length=32):
    t = torch.linspace(0, 1, length)
    return torch.stack([torch.sin(14 * torch.pi * t) + t,
                        torch.cos(8 * torch.pi * t) - .3 * t], -1)[None]


def test_real_ceemdan_matches_backend_and_noise_reuse():
    cfg = CEEMDANConfig(trials=2, max_imf=2, max_sift=10)
    module = CEEMDANDecomposition(cfg)
    for x in [signal().double(), signal().double() * .7 + .3]:
        oscillation, remainder = module(x)
        np.testing.assert_allclose((oscillation + remainder).numpy(), x.numpy(), atol=1e-12)
        for d in range(2):
            ce = CEEMDAN(trials=cfg.trials, epsilon=cfg.epsilon,
                         ext_EMD=EMD(MAX_ITERATION=cfg.max_sift), parallel=False)
            ce.noise_seed(cfg.seed)
            modes = ce.ceemdan(x[0, :, d].numpy(), max_imf=cfg.max_imf)
            np.testing.assert_allclose(oscillation[0, :, d], modes[:-1].sum(0), atol=1e-12)


def test_stop_gradient_convention_is_explicit():
    x = signal().requires_grad_()
    seasonal, trend = CEEMDANDecomposition(CEEMDANConfig(trials=2, max_imf=1))(x)
    assert not trend.requires_grad
    seasonal.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('kind', ['constant', 'linear', 'short'])
def test_degenerate_series(kind):
    x = torch.ones(2, 9, 2)
    if kind == 'linear':
        x = torch.arange(9.).expand(2, 2, -1).transpose(1, 2)
    if kind == 'short':
        x = signal(3)
    seasonal, trend = CEEMDANDecomposition(CEEMDANConfig(trials=2, max_imf=1))(x)
    assert torch.isfinite(trend).all()
    torch.testing.assert_close(seasonal + trend, x)
    if kind != 'short':
        torch.testing.assert_close(seasonal, torch.zeros_like(x))


def test_batch_and_feature_permutation_do_not_change_decomposition():
    module = CEEMDANDecomposition(CEEMDANConfig(trials=2, max_imf=1))
    x = torch.cat([signal(), signal() * .7], 0)
    _, expected = module(x)
    _, permuted = module(x.flip(0).flip(2))
    torch.testing.assert_close(expected, permuted.flip(0).flip(2))


def test_every_layer_decomposes_twice_and_all_paths_train():
    torch.manual_seed(42)
    model = Model(config()).train()
    counts = []
    handles = []
    for layer in model.base.backbone.encoder.layers:
        for module in [layer.decomp1, layer.decomp2]:
            handles.append(module.register_forward_hook(lambda *args: counts.append(1)))
    prediction, parts = model(signal(), return_components=True)
    assert prediction.shape == (1, 8, 2)
    assert len(counts) == 4
    assert len(parts['layer_trends_normalized']) == 2
    prediction.square().mean().backward()
    for name in ['base.backbone.W_P.weight',
                 'base.backbone.encoder.layers.0.self_attn.W_Q.weight',
                 'base.backbone.encoder.layers.0.autocorrelation.v.weight',
                 'base.backbone.encoder.layers.0.mix_logit',
                 'base.backbone.encoder.layers.0.ff.0.weight',
                 'base.head.linear.weight', 'trend_heads.0.1.weight', 'trend_heads.1.1.weight']:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().max() > 0, name
    for handle in handles:
        handle.remove()


def test_channel_independence_and_batch_invariance():
    torch.manual_seed(11)
    model = Model(config()).eval()
    x = signal()
    with torch.no_grad():
        expected = model(x)
        changed = x.clone()
        changed[:, :, 1] = 13 * changed[:, :, 1] + 50
        actual = model(changed)
        torch.testing.assert_close(actual[:, :, 0], expected[:, :, 0], atol=1e-6, rtol=1e-6)
        larger_batch = model(torch.cat([x, changed], 0))
        torch.testing.assert_close(larger_batch[:1], expected, atol=1e-6, rtol=1e-6)


def test_autocorrelation_odd_length_matches_manual_delay_aggregation():
    torch.manual_seed(3)
    module = InstanceAutoCorrelation(4, 2, factor=100).eval()
    x = torch.randn(2, 7, 4)
    split = lambda z: z.reshape(2, 7, 2, 2).permute(0, 2, 3, 1)
    q, k, v = split(module.q(x)), split(module.k(x)), split(module.v(x))
    # Direct circular correlation checks the FFT and delay sign conventions.
    corr = torch.stack([(q * torch.roll(k, i, -1)).sum(-1).mean((1, 2))
                        for i in range(7)], -1)
    weights = corr.softmax(-1)
    expected = sum(torch.roll(v, -i, -1) * weights[:, i, None, None, None] for i in range(7))
    expected = module.out(expected.permute(0, 3, 1, 2).reshape(2, 7, 4))
    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize('decomp', ['ceemdan', 'moving_avg', 'none'])
@pytest.mark.parametrize('attention', ['self', 'hybrid', 'autocorrelation'])
def test_ablations_and_state_dict_reload(decomp, attention):
    model = Model(config(e_layers=1, progressive_decomp=decomp, fusion_attention=attention)).eval()
    restored = Model(config(e_layers=1, progressive_decomp=decomp, fusion_attention=attention)).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = model(signal())
        torch.testing.assert_close(restored(signal()), expected)
    assert torch.isfinite(expected).all()


def test_normalized_components_reconstruct_prediction():
    model = Model(config(e_layers=1)).eval()
    with torch.no_grad():
        prediction, parts = model(signal(), return_components=True)
        normalized = parts['seasonal_normalized'] + parts['trend_normalized']
        torch.testing.assert_close(model.base.revin_layer(normalized, 'denorm'), prediction)
        torch.testing.assert_close(sum(parts['layer_trends_normalized']), parts['trend_normalized'])


def test_nonfinite_input_rejected():
    x = signal()
    x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        Model(config())(x)


def test_no_decomp_self_attention_matches_official_patchtst_with_same_weights():
    # The upstream wrapper does not forward its norm option to TSTiEncoder;
    # compare its actual default BatchNorm architecture without modifying it.
    cfg = config(e_layers=1, encoder_norm='BatchNorm', fusion_attention='self', progressive_decomp='none')
    official = OfficialPatchTST(cfg).eval()
    fused = Model(cfg).eval()
    missing, unexpected = fused.base.load_state_dict(official.model.state_dict(), strict=False)
    assert missing == [] and unexpected == []
    with torch.no_grad():
        torch.testing.assert_close(fused(signal()), official(signal()))


def test_layernorm_channel_independence_in_training_mode():
    model = Model(config(e_layers=1)).train()
    x = signal()
    changed = x.clone()
    changed[:, :, 1] = torch.randn_like(changed[:, :, 1])
    torch.testing.assert_close(model(x)[:, :, 0], model(changed)[:, :, 0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('revin', [False, True])
@pytest.mark.parametrize('padding_patch', [None, 'end'])
def test_official_options(revin, padding_patch):
    model = Model(config(e_layers=1, revin=revin, padding_patch=padding_patch,
                         subtract_last=True, affine=False)).eval()
    with torch.no_grad():
        y = model(signal())
    assert y.shape == (1, 8, 2) and torch.isfinite(y).all()
