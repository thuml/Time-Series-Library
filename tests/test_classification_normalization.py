import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_provider.data_factory import data_provider
from data_provider.data_loader import UEAloader
from exp.exp_classification import Exp_Classification


class ClassificationNormalizationTest(unittest.TestCase):
    def test_local_ts_files_reach_test_batches_without_refitting(self):
        header = (
            '@problemName Synthetic\n@timestamps false\n@univariate true\n'
            '@classLabel true a b\n@data\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Synthetic_TRAIN.ts').write_text(
                header + '1,2:a\n3,4:b\n', encoding='utf-8',
            )
            (root / 'Synthetic_TEST.ts').write_text(
                header + '101,102:a\n103,104:b\n', encoding='utf-8',
            )
            experiment = Exp_Classification.__new__(Exp_Classification)
            experiment.normalizer = None
            experiment.args = SimpleNamespace(
                data='UEA', embed='timeF', batch_size=2, freq='h',
                task_name='classification', root_path=str(root), num_workers=0,
                seq_len=2, model_id='Synthetic', augmentation_ratio=0,
            )
            train_data, _ = experiment._get_data('TRAIN')
            mean = train_data.normalizer.mean.copy()
            std = train_data.normalizer.std.copy()
            test_data, test_loader = experiment._get_data('TEST')
            batch, labels, mask = next(iter(test_loader))

            # Independent oracle: training mean 2.5, sample std sqrt(5/3).
            expected = (np.array([101., 102., 103., 104.]) - 2.5) / np.sqrt(5 / 3)
            np.testing.assert_allclose(batch.numpy().reshape(-1), expected, rtol=1e-6)
            np.testing.assert_array_equal(labels.numpy().reshape(-1), [0, 1])
            self.assertTrue(mask.numpy().all())
            self.assertIs(test_data.normalizer, train_data.normalizer)
            pd.testing.assert_series_equal(train_data.normalizer.mean, mean)
            pd.testing.assert_series_equal(train_data.normalizer.std, std)

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
