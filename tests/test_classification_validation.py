from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from data_provider import data_factory
from exp.exp_classification import Exp_Classification


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([
            [3.0, 0.0, -2.0],
            [-1.0, 2.0, 0.5],
            [0.0, 0.2, 0.8],
            [1.0, 1.2, 0.0],
            [4.0, 1.0, 0.0],
        ]))
        self.forward_modes = []

    def forward(self, features, padding_mask, *_):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return self.logits[features[:, 0, 0].long()]


@pytest.mark.parametrize("batch_size,targets,batch_lengths", [
    (1, [0], [1]),
    (2, [0, 1, 2, 1], [2, 2]),
    (4, [0, 2, 2, 2], [4]),
    (2, [0, 2, 2, 1, 0], [2, 2, 1]),
    (3, [0, 2, 2, 1, 0], [3, 2]),
    (1, [2, 2, 1], [1, 1, 1]),
], ids=["singleton", "full-batches", "larger-batch", "final-singleton",
        "final-short-batch", "consecutive-singletons"])
def test_classification_validation(monkeypatch, batch_size, targets, batch_lengths):
    samples = [
        (torch.full((1 + index % 3, 1), float(index)), torch.tensor([target]))
        for index, target in enumerate(targets)
    ]
    monkeypatch.setitem(data_factory.data_dict, "UEA", lambda **kwargs: samples)

    # Skip model construction and dataset reads. Exercise the real data provider and validator.
    experiment = Exp_Classification.__new__(Exp_Classification)
    experiment.device = torch.device("cpu")
    experiment.model = FixedLogits()
    experiment.args = SimpleNamespace(
        data="UEA", task_name="classification", embed="", freq="h", root_path="",
        batch_size=batch_size, seq_len=3, num_workers=0,
    )
    dataset, loader = experiment._get_data("TEST")
    assert loader.drop_last is False
    batches = list(loader)
    assert [labels.shape for _, labels, _ in batches] == [(n, 1) for n in batch_lengths]
    original_batches = [tuple(tensor.clone() for tensor in batch) for batch in batches]

    criterion = experiment._select_criterion()
    criterion_shapes = []

    def record_target_shape(module, inputs):
        logits, labels = inputs
        criterion_shapes.append((logits.shape, labels.shape, labels.dtype))

    criterion.register_forward_pre_hook(record_target_shape)
    original_logits = experiment.model.logits.detach().clone()
    assert experiment.model.training
    loss, accuracy = experiment.vali(dataset, batches, criterion)

    expected_batch_losses = []
    true_labels = torch.tensor(targets)
    logits = original_logits[:len(targets)]
    for start in range(0, len(targets), batch_size):
        batch = slice(start, start + batch_size)
        expected_batch_losses.append(F.cross_entropy(logits[batch], true_labels[batch]).item())
    assert isinstance(loss, np.floating)
    assert isinstance(accuracy, np.floating)
    assert loss == pytest.approx(sum(expected_batch_losses) / len(expected_batch_losses))
    assert accuracy == pytest.approx(float((logits.argmax(dim=1) == true_labels).sum()) / len(targets))
    assert criterion_shapes == [((n, 3), (n,), torch.long) for n in batch_lengths]

    assert experiment.model.forward_modes == [(False, False)] * len(batch_lengths)
    assert experiment.model.training
    assert experiment.model.logits.grad is None
    torch.testing.assert_close(experiment.model.logits, original_logits)
    for batch, original in zip(batches, original_batches):
        for tensor, original_tensor in zip(batch, original):
            torch.testing.assert_close(tensor, original_tensor)
