import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_provider.data_factory import data_provider
from data_provider.data_loader import UEAloader
from exp.exp_classification import Exp_Classification


class ClassificationNormalizationTest(unittest.TestCase):
    def test_test_split_uses_training_statistics(self):
        train_features = pd.DataFrame(
            {'feature': [1.0, 2.0, 3.0, 4.0]},
            index=[0, 0, 1, 1],
        )
        test_features = pd.DataFrame(
            {'feature': [101.0, 102.0, 103.0, 104.0]},
            index=[0, 0, 1, 1],
        )
        labels = pd.DataFrame([0, 1], dtype=np.int8)

        def load_all(_loader, _root_path, file_list=None, flag=None):
            del file_list
            features = train_features if flag == 'TRAIN' else test_features
            return features.copy(), labels.copy()

        args = SimpleNamespace(
            data='UEA',
            embed='timeF',
            batch_size=2,
            freq='h',
            task_name='classification',
            root_path='.',
            num_workers=0,
            seq_len=2,
            model_id='Synthetic',
            augmentation_ratio=0,
        )
        with patch.object(UEAloader, 'load_all', new=load_all):
            train_data, _ = data_provider(args, 'TRAIN')
            test_data, _ = data_provider(
                args,
                'TEST',
                normalizer=train_data.normalizer,
            )

        expected = (test_features - train_features.mean()) / (
            train_features.std() + np.finfo(float).eps
        )
        self.assertIs(test_data.normalizer, train_data.normalizer)
        np.testing.assert_allclose(test_data.feature_df, expected)
        self.assertGreater(abs(test_data.feature_df.mean().iloc[0]), 1.0)

    def test_experiment_passes_training_normalizer_to_test_split(self):
        experiment = Exp_Classification.__new__(Exp_Classification)
        experiment.args = SimpleNamespace()
        experiment.normalizer = None
        training_normalizer = object()
        calls = []

        def provide_data(_args, flag, normalizer=None):
            calls.append((flag, normalizer))
            data_set = SimpleNamespace(normalizer=training_normalizer)
            return data_set, object()

        with patch('exp.exp_classification.data_provider', new=provide_data):
            experiment._get_data('TRAIN')
            experiment._get_data('TEST')

        self.assertEqual(calls[0], ('TRAIN', None))
        self.assertEqual(calls[1], ('TEST', training_normalizer))


if __name__ == '__main__':
    unittest.main()
