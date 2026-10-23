"""Audit original-target predictions and quantify paired uncertainty."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from label_denoise_experiment import VARIANTS, error_metrics, model_config
from models.PatchTST import Model
from quick_validate import atomic_json


def paired_interval(base, candidate, seed=20261022, trials=1000, block=96):
    """Resample paired seeds and circular 96-origin time blocks together."""
    rng = np.random.default_rng(seed)
    n_seeds, n_origins = base.shape
    changes = []
    for _ in range(trials):
        seeds = rng.integers(0, n_seeds, n_seeds)
        starts = rng.integers(0, n_origins, math.ceil(n_origins / block))
        indices = ((starts[:, None] + np.arange(block)) % n_origins).ravel()[:n_origins]
        b = base[seeds][:, indices].mean()
        c = candidate[seeds][:, indices].mean()
        changes.append((c / b - 1) * 100)
    return dict(relative_mse_change_percent=float((candidate.mean()/base.mean()-1)*100),
                interval95_percent=np.percentile(changes, [2.5, 97.5]).tolist(),
                bootstrap_trials=trials, time_block_hours=block,
                method='Paired seed and circular time-block bootstrap; approximate pilot uncertainty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    out = args.experiment
    assert json.loads((out/'status.json').read_text())['status'] == 'complete'
    protocol = json.loads((out/'protocol.json').read_text())
    records = json.loads((out/'results.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    cfg = protocol['config']
    torch.set_num_threads(1)
    assert hashlib.sha256(Path(cfg['data_path']).read_bytes()).hexdigest() == protocol['data_sha256']
    assert hashlib.sha256(Path('label_denoise_experiment.py').read_bytes()).hexdigest() == protocol['source_sha256']
    df = pd.read_csv(cfg['data_path'])
    raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    with np.load(out/'sample_origins.npz') as sample:
        origins = sample['test']
        mean, scale = sample['train_mean'], sample['train_std']
        assert sample['train'].max() + cfg['pred_len'] <= cfg['train_end']
        assert sample['validation'].min() == cfg['train_end']
        assert origins.min() == cfg['validation_end']
    normalized = ((raw-mean)/scale).astype(np.float32)
    true = np.load(out/'test_true.npy')
    expected = np.stack([normalized[t:t+cfg['pred_len']] for t in origins])
    np.testing.assert_array_equal(expected, true)
    with np.load(out/'training_labels.npz') as labels:
        np.testing.assert_array_equal(labels['raw'], raw[:cfg['train_end']])
        np.testing.assert_allclose(labels['drop_imf1']+labels['imf1'], labels['raw'], atol=1e-10)
        np.testing.assert_allclose(labels['drop_imf1_imf2']+labels['imf1']+labels['imf2'], labels['raw'], atol=1e-10)
    per_origin = {v: [] for v in VARIANTS}
    reload_differences = []
    for record in records:
        folder = out/'models'/f'{record["variant"]}_{record["seed"]}'
        pred = np.load(folder/'test_pred.npy')
        measured = error_metrics(pred, true, scale)
        np.testing.assert_allclose([measured['mse'], measured['mae']],
            [record['test']['mse'], record['test']['mae']], rtol=1e-12, atol=1e-12)
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == record['best_epoch']
        assert checkpoint['seed'] == record['seed'] and checkpoint['variant'] == record['variant']
        model = Model(model_config(cfg, len(protocol['columns']))).eval()
        model.load_state_dict(checkpoint['model'])
        # DataLoader's collate produces contiguous tensors; reproduce that
        # layout so CPU reduction kernels match the original evaluation.
        batch = torch.from_numpy(np.stack([normalized[t-cfg['seq_len']:t] for t in origins[:cfg['batch_size']]])).contiguous()
        with torch.no_grad():
            restored = model(batch).numpy()
        np.testing.assert_allclose(restored, pred[:len(batch)], rtol=1e-6, atol=1e-6)
        reload_differences.append(float(np.max(abs(restored-pred[:len(batch)]))))
        e = pred.astype(np.float64)-true.astype(np.float64)
        per_origin[record['variant']].append(np.mean(e**2, axis=(1, 2)))
    for seed in cfg['seeds']:
        assert len(set(r['initial_state_sha256'] for r in records if r['seed']==seed)) == 1
    for v in VARIANTS:
        per_origin[v] = np.stack(per_origin[v])
    uncertainty = {v: paired_interval(per_origin['raw'], per_origin[v]) for v in VARIANTS[1:]}
    atomic_json(out/'paired_uncertainty.json', uncertainty)
    atomic_json(out/'verification.json', dict(status='passed', raw_test_labels_verified=True,
        decomposition_train_only=True, label_reconstruction_verified=True,
        paired_initializations_verified=True, saved_metrics_reproduced=True,
        checkpoints_independently_reload=True, models_verified=len(records),
        test_windows=len(origins), experiment_source_unchanged=True,
        maximum_reload_absolute_difference=max(reload_differences)))
    names = dict(raw='原始 label', drop_imf1='删除 IMF1', drop_imf1_imf2='删除 IMF1＋IMF2')
    note = '\n\n## 配对不确定性与核对\n\n'
    for v, measured in uncertainty.items():
        lo, hi = measured['interval95_percent']
        note += (f'- {names[v]}：测试 MSE 相对变化 {measured["relative_mse_change_percent"]:+.2f}%，'
                 f'配对区块自举 95% 区间 [{lo:+.2f}%, {hi:+.2f}%]。\n')
    note += ('\n自举同时重采样 3 个种子及 96 小时的环形时间区块，缓解重叠预测窗口的相关性；'
             '这只是本小样本试验的近似不确定性估计。所有测试标签、误差重算、'
             '训练段分解重构、同种子初始化与 9 份检查点重载均核对通过，详见 verification.json。\n')
    with (out/'REPORT.zh.md').open('a') as f:
        f.write(note)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    with np.load(out/'training_labels.npz') as labels:
        begin, end = 2160, 2328
        c = protocol['columns'].index('OT')
        for key, name, style in [('raw', 'Raw label', '-'), ('drop_imf1','Drop IMF1','--'),
                                  ('drop_imf1_imf2','Drop IMF1 + IMF2',':')]:
            axes[0].plot(np.arange(end-begin), labels[key][begin:end, c], style,
                         label=name, linewidth=1.3, alpha=.9)
    axes[0].set(xlabel='Hours in one training week', ylabel='OT (original units)',
                title='Only the training label changes')
    axes[0].legend(fontsize=8)
    labels_short = ['Raw', 'Drop IMF1', 'Drop IMF1+2']
    means = [summary['test'][v]['mse']['mean'] for v in VARIANTS]
    deviations = [summary['test'][v]['mse']['std'] for v in VARIANTS]
    axes[1].errorbar(range(3), means, yerr=deviations, fmt='o', capsize=5, color='#2368a0')
    for i, v in enumerate(VARIANTS):
        axes[1].scatter(np.full(3, i)+np.linspace(-.07,.07,3), summary['test'][v]['mse']['seeds'],
                         color='#4d4d4d', alpha=.65, s=18)
    axes[1].set_xticks(range(3), labels_short)
    axes[1].set(ylabel='Test MSE against raw targets', title='Mean and seed standard deviation')
    axes[1].grid(axis='y', alpha=.2)
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(verification='passed', uncertainty=uncertainty,
                         summary=summary),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
