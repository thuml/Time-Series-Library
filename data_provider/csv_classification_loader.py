import glob
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVMultiFileClassificationLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = (flag or "TRAIN").upper()
        self.seq_len = args.seq_len
        self.window_step = max(1, int(getattr(args, "window_step", 1)))
        self.label_col = getattr(args, "label_col", "label")
        self.window_label_mode = getattr(args, "window_label_mode", "last").lower()
        self.train_ratio = float(getattr(args, "train_ratio", 0.7))
        self.val_ratio = float(getattr(args, "val_ratio", 0.15))
        self.file_split_mode = getattr(args, "file_split_mode", "shuffle").lower()
        self.drop_cols = {
            col.strip() for col in str(getattr(args, "drop_cols", "")).split(",") if col.strip()
        }
        self.sampler_power = float(getattr(args, "sampler_power", 1.0))
        self.minority_boost = float(getattr(args, "minority_boost", 1.0))
        self.minority_raw_label = self._parse_optional_label(getattr(args, "minority_raw_label", ""))
        self._file_cache = {}

        self.csv_files = self._list_csv_files()
        self.split_files = self._split_files(self.csv_files)
        self.class_names = self._collect_class_names(self.csv_files)
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        sample_features, _ = self._load_file(self.csv_files[0])
        self.feature_names = list(sample_features.columns)
        self.feature_df = pd.DataFrame(columns=self.feature_names)
        self.max_seq_len = self.seq_len

        self.train_mean, self.train_std = self._fit_normalizer(self.split_files["TRAIN"])
        self.class_weights = self._compute_class_weights(self.split_files["TRAIN"])
        self.samples, label_dist = self._build_samples(self.split_files[self.flag])
        self.sample_weights = self._compute_sample_weights(self.samples)
        self.sample_metadata = [
            {
                "file_name": sample["file_name"],
                "start_idx": sample["start_idx"],
                "end_idx": sample["end_idx"],
                "raw_label": sample["raw_label"],
            }
            for sample in self.samples
        ]

        print(
            f"{self.flag}: files={len(self.split_files[self.flag])}, "
            f"windows={len(self.samples)}, features={len(self.feature_names)}, "
            f"label_dist={dict(sorted(label_dist.items(), key=lambda item: item[0]))}"
        )
        if self.flag == "TRAIN" and self.minority_raw_label is not None and self.minority_raw_label not in self.class_names:
            print(f"Warning: minority_raw_label={self.minority_raw_label} not found in train classes {self.class_names}")

    @staticmethod
    def _parse_optional_label(raw_value):
        if raw_value is None:
            return None
        raw_text = str(raw_value).strip()
        if not raw_text or raw_text.lower() in {"none", "null", "nan"}:
            return None
        try:
            return int(raw_text)
        except ValueError:
            pass
        try:
            return float(raw_text)
        except ValueError:
            return raw_text

    def _list_csv_files(self):
        pattern = os.path.join(self.root_path, "**", "*.csv")
        csv_files = sorted(glob.glob(pattern, recursive=True))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {self.root_path}")
        return csv_files

    def _split_files(self, files):
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")

        split_files = list(files)
        if self.file_split_mode == "shuffle":
            random.Random(getattr(self.args, "seed", 2)).shuffle(split_files)
        elif self.file_split_mode != "sorted":
            raise ValueError(
                f"Unsupported file_split_mode={self.file_split_mode}. Use 'shuffle' or 'sorted'."
            )

        total_files = len(split_files)
        train_end = int(total_files * self.train_ratio)
        val_end = int(total_files * (self.train_ratio + self.val_ratio))
        train_files = split_files[:train_end]
        val_files = split_files[train_end:val_end]
        test_files = split_files[val_end:]

        if not train_files or not val_files or not test_files:
            raise ValueError(
                f"Invalid file split produced empty subset: "
                f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
            )

        return {"TRAIN": train_files, "VAL": val_files, "TEST": test_files}

    def _prepare_feature_frame(self, df):
        if self.label_col not in df.columns:
            raise KeyError(f"Label column '{self.label_col}' not found in CSV columns: {list(df.columns)}")

        feature_cols = [col for col in df.columns if col not in self.drop_cols and col != self.label_col]
        feature_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        invalid_cols = [
            col for col in feature_cols
            if feature_df[col].isna().all() and df[col].notna().any()
        ]
        if invalid_cols:
            raise ValueError(
                f"Non-numeric feature columns detected: {invalid_cols}. "
                f"Add them to --drop_cols."
            )

        feature_df = feature_df.ffill().bfill().fillna(0.0)
        return feature_df

    def _load_file(self, file_path):
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        df = pd.read_csv(file_path)
        feature_df = self._prepare_feature_frame(df)
        label_series = pd.to_numeric(df[self.label_col], errors="raise")
        if np.allclose(label_series.values, label_series.values.astype(int)):
            label_series = label_series.astype(int)

        self._file_cache[file_path] = (feature_df, label_series)
        return self._file_cache[file_path]

    def _collect_class_names(self, files):
        label_values = set()
        for file_path in files:
            _, label_series = self._load_file(file_path)
            label_values.update(label_series.dropna().tolist())
        return sorted(label_values)

    def _fit_normalizer(self, train_files):
        train_features = []
        for file_path in train_files:
            feature_df, _ = self._load_file(file_path)
            train_features.append(feature_df)

        concat_features = pd.concat(train_features, axis=0, ignore_index=True)
        mean = concat_features.mean()
        std = concat_features.std().replace(0, 1.0).fillna(1.0)
        return mean, std

    def _normalize_features(self, feature_df):
        return (feature_df - self.train_mean) / (self.train_std + np.finfo(float).eps)

    def _window_label(self, labels_window):
        if self.window_label_mode == "last":
            return labels_window[-1]
        if self.window_label_mode == "majority":
            return Counter(labels_window).most_common(1)[0][0]
        raise ValueError(
            f"Unsupported window_label_mode={self.window_label_mode}. Use 'last' or 'majority'."
        )

    def _build_samples(self, files):
        samples = []
        label_dist = Counter()

        for file_path in files:
            feature_df, label_series = self._load_file(file_path)
            feature_df = self._normalize_features(feature_df)
            feature_values = feature_df.to_numpy(dtype=np.float32, copy=True)
            label_values = label_series.to_numpy(copy=True)

            if len(feature_values) < self.seq_len:
                continue

            for start_idx in range(0, len(feature_values) - self.seq_len + 1, self.window_step):
                end_idx = start_idx + self.seq_len
                raw_label = self._window_label(label_values[start_idx:end_idx])
                label_idx = self.label_to_index[raw_label]
                samples.append(
                    {
                        "x": np.ascontiguousarray(feature_values[start_idx:end_idx]),
                        "label": label_idx,
                        "raw_label": raw_label,
                        "file_name": os.path.basename(file_path),
                        "start_idx": start_idx,
                        "end_idx": end_idx - 1,
                    }
                )
                label_dist[raw_label] += 1

        if not samples:
            raise ValueError(
                f"No valid sliding-window samples were created for split {self.flag}. "
                f"Check seq_len={self.seq_len} and the CSV contents."
            )

        return samples, label_dist

    def _compute_class_weights(self, train_files):
        class_counter = Counter()
        for file_path in train_files:
            _, label_series = self._load_file(file_path)
            label_values = label_series.to_numpy(copy=True)
            if len(label_values) < self.seq_len:
                continue
            for start_idx in range(0, len(label_values) - self.seq_len + 1, self.window_step):
                end_idx = start_idx + self.seq_len
                raw_label = self._window_label(label_values[start_idx:end_idx])
                class_counter[self.label_to_index[raw_label]] += 1

        total = sum(class_counter.values())
        num_classes = len(self.class_names)
        weights = []
        for class_idx in range(num_classes):
            count = class_counter.get(class_idx, 0)
            weight = total / (num_classes * count) if count > 0 else 0.0
            weights.append(weight)
        return np.asarray(weights, dtype=np.float32)

    def _compute_sample_weights(self, samples):
        if len(self.class_weights) == 0:
            return np.ones(len(samples), dtype=np.float64)

        sample_weights = []
        for sample in samples:
            label_idx = int(sample["label"])
            base_weight = float(self.class_weights[label_idx])
            weight = max(base_weight, np.finfo(np.float32).eps) ** self.sampler_power
            if self.minority_raw_label is not None and sample["raw_label"] == self.minority_raw_label:
                weight *= self.minority_boost
            sample_weights.append(weight)
        return np.asarray(sample_weights, dtype=np.float64)

    def decode_indices(self, indices):
        return [self.index_to_label[int(index)] for index in indices]

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            torch.from_numpy(sample["x"]),
            torch.tensor([sample["label"]], dtype=torch.long),
        )

    def __len__(self):
        return len(self.samples)
