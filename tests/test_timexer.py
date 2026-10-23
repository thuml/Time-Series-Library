from types import SimpleNamespace

import pytest
import torch

from models.TimeXer import Model


def config(**overrides):
    values = dict(seq_len=32, pred_len=8, features='M', enc_in=3, d_model=8,
                  n_heads=2, d_ff=16, e_layers=2, dropout=0., patch_len=8,
                  factor=1, activation='gelu', embed='timeF', freq='h',
                  use_norm=1, output_attention=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('features,channels', [('M', 3), ('MS', 3), ('S', 1)])
@pytest.mark.parametrize('use_norm', [0, 1])
def test_forecast_gradients_and_checkpoint(features, channels, use_norm, tmp_path):
    torch.manual_seed(42)
    cfg = config(features=features, enc_in=channels, use_norm=use_norm)
    model = Model(cfg)
    x = torch.randn(2, 32, channels, requires_grad=True)
    marks = torch.randn(2, 32, 4)
    output = model(x, marks)
    assert output.shape == (2, 8, channels if features == 'M' else 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.en_embedding.glb_token.grad.abs().sum() > 0
    assert model.ex_embedding.value_embedding.weight.grad.abs().sum() > 0
    for layer in model.encoder.layers:
        assert layer.cross_attention.query_projection.weight.grad.abs().sum() > 0
    if features == 'MS':
        assert x.grad[:, :, :-1].abs().sum() > 0
    checkpoint = tmp_path / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    reloaded = Model(cfg).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(output, reloaded(x, marks))


def test_ms_embeds_target_separately_and_ignores_future_values():
    model = Model(config(features='MS')).eval()
    x, marks = torch.randn(2, 32, 3), torch.randn(2, 32, 4)
    seen = {}
    def endogenous(_, args):
        seen['endogenous'] = args[0]
    def exogenous(_, args):
        seen['exogenous'] = args[0]
    model.en_embedding.register_forward_pre_hook(endogenous)
    model.ex_embedding.register_forward_pre_hook(exogenous)
    a = model(x, marks, torch.zeros(2, 8, 3), torch.zeros(2, 8, 4))
    b = model(x, marks, torch.randn(2, 8, 3), torch.randn(2, 8, 4))
    torch.testing.assert_close(a, b)
    assert seen['endogenous'].shape == (2, 1, 32)
    assert seen['exogenous'].shape == (2, 32, 2)
    perturbed = x.clone()
    perturbed[:, :16, 0] += 10
    assert not torch.allclose(a, model(perturbed, marks))


def test_cross_attention_uses_only_global_tokens():
    model = Model(config()).eval()
    shapes = []
    def inspect(_, args):
        shapes.append((args[0].shape, args[1].shape))
    model.encoder.layers[0].cross_attention.register_forward_pre_hook(inspect)
    model(torch.randn(2, 32, 3), torch.randn(2, 32, 4))
    assert shapes == [(torch.Size([2, 3, 8]), torch.Size([2, 7, 8]))]


def test_partial_patch_and_attention_runner_contract():
    model = Model(config(seq_len=35, output_attention=True))
    output, attention = model(torch.randn(1, 35, 3))
    assert output.shape == (1, 8, 3)
    assert attention is None  # official TimeXer disables attention-map collection


@pytest.mark.parametrize('overrides', [dict(patch_len=33), dict(patch_len=0),
                         dict(features='S'), dict(d_model=7), dict(e_layers=0)])
def test_invalid_configs(overrides):
    with pytest.raises(ValueError):
        Model(config(**overrides))


def test_single_channel_requires_cross_tokens():
    with pytest.raises(ValueError, match='time markers'):
        Model(config(features='S', enc_in=1))(torch.randn(2, 32, 1))


def test_selected_ms_targets_exclude_all_targets_from_exogenous_context():
    torch.manual_seed(2021)
    model = Model(config(features='MS', enc_in=6, target_indices=[0, 3])).eval()
    x = torch.randn(2, 32, 6, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 8, 2)
    assert model.exogenous_indices.tolist() == [1, 2, 4, 5]
    output.square().mean().backward()
    assert x.grad[:, :, [1, 2, 4, 5]].abs().sum() > 0
    perturbed = x.detach().clone()
    perturbed[:, :16, 0] += 5
    changed = model(perturbed)
    torch.testing.assert_close(output[:, :, 1], changed[:, :, 1])
    assert not torch.allclose(output[:, :, 0], changed[:, :, 0])


def test_selected_single_ms_matches_official_target_last_reordering():
    explicit = Model(config(features='MS', enc_in=4, target_indices=[1])).eval()
    official = Model(config(features='MS', enc_in=4)).eval()
    official.load_state_dict(explicit.state_dict(), strict=True)
    x = torch.randn(2, 32, 4)
    torch.testing.assert_close(explicit(x), official(x[:, :, [0, 2, 3, 1]]))


@pytest.mark.parametrize('targets', [[], [0, 0], [-1], [3], [True]])
def test_invalid_target_selection(targets):
    with pytest.raises(ValueError, match='target_indices'):
        Model(config(features='MS', target_indices=targets))
