from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom


@pytest.mark.parametrize('features,channels', [('M', 3), ('MS', 3), ('S', 1)])
def test_traffic_splits_scaling_target_and_factory(tmp_path, features, channels):
    frame = pd.DataFrame(dict(date=pd.date_range('2015-01-01', periods=200, freq='h'),
                              OT=np.arange(200) * 3., a=np.arange(200) * 1.,
                              b=np.arange(200) * 2.))
    frame.to_csv(tmp_path / 'traffic.csv', index=False)
    args = SimpleNamespace(data='traffic', embed='timeF', batch_size=7, freq='h',
                           root_path=str(tmp_path), data_path='traffic.csv',
                           seq_len=16, label_len=8, pred_len=8, features=features,
                           target='OT', num_workers=0)
    expected = {'train': (117, 0), 'val': (13, 124), 'test': (33, 144)}
    for flag, (windows, start) in expected.items():
        dataset, loader = data_provider(args, flag)
        assert isinstance(dataset, Dataset_Custom)
        assert len(dataset) == windows
        x, y, xm, ym = dataset[0]
        assert x.shape == (16, channels) and y.shape == (16, channels)
        assert xm.shape == ym.shape == (16, 4)
        assert dataset.scaler.mean_[-1] == frame.OT.iloc[:140].mean()
        original = dataset.inverse_transform(y)
        np.testing.assert_allclose(original[:, -1], frame.OT.iloc[start + 8:start + 24])
        if flag != 'train':
            assert not loader.drop_last
            assert sum(len(batch[0]) for batch in loader) == windows
