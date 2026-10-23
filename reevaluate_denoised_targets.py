"""Post-hoc, common-reference evaluation of frozen label-denoising models.

No retraining or checkpoint reselection. The held-out test block is filtered
offline solely to define retrospective evaluation references.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd

from label_denoise_experiment import VARIANTS, denoise_channel
from quick_validate import atomic_json


def percentage_metrics(pred, target, epsilon):
    """Original-unit MAPE and WMAPE; zero targets remain in WMAPE error sums.

    Primary aggregates are equal-weight means of per-variable metrics, avoiding
    unit-dependent mixing of temperature and load. Pooled WMAPE is also saved.
    """
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError('Expected matching [windows, horizon, channels]')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite percentage-metric inputs')
    epsilon = np.broadcast_to(np.asarray(epsilon, dtype=float), (target.shape[-1],))
    if not np.isfinite(epsilon).all() or (epsilon < 0).any():
        raise ValueError('epsilon must be finite and nonnegative')
    error, denominator = abs(pred-target), abs(target)
    mape, wmape, excluded = [], [], []
    for c in range(target.shape[-1]):
        valid = denominator[..., c] > epsilon[c]
        excluded.append(int(valid.size - valid.sum()))
        mape.append(float(100*np.mean(error[..., c][valid]/denominator[..., c][valid]))
                    if valid.any() else None)
        total = float(denominator[..., c].sum())
        wmape.append(float(100*error[..., c].sum()/total) if total > 0 else None)
    safe_mean = lambda a: float(np.mean([x for x in a if x is not None])) if any(x is not None for x in a) else None
    total = float(denominator.sum())
    return dict(mape_macro_percent=safe_mean(mape), wmape_macro_percent=safe_mean(wmape),
                wmape_pooled_percent=float(100*error.sum()/total) if total > 0 else None,
                per_channel_mape_percent=mape, per_channel_wmape_percent=wmape,
                mape_excluded_per_channel=excluded,
                observations_per_channel=int(np.prod(target.shape[:-1])))


def scores(pred_scaled, target_scaled, pred_physical, target_physical, epsilon):
    e = pred_scaled.astype(np.float64)-target_scaled.astype(np.float64)
    result = percentage_metrics(pred_physical, target_physical, epsilon)
    result.update(mse=float(np.mean(e**2)), mae=float(np.mean(abs(e))),
                  per_channel_mse=np.mean(e**2, axis=(0, 1)).tolist(),
                  per_channel_mae=np.mean(abs(e), axis=(0, 1)).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = args.experiment
    previous_protocol = json.loads((source/'protocol.json').read_text())
    cfg = previous_protocol['config']
    records = json.loads((source/'results.json').read_text())
    raw = pd.read_csv(cfg['data_path']).iloc[:, 1:].to_numpy(dtype=np.float64)
    if hashlib.sha256(Path(cfg['data_path']).read_bytes()).hexdigest() != previous_protocol['data_sha256']:
        raise ValueError('Original data changed')
    with np.load(source/'sample_origins.npz') as a:
        origins = a['test'].copy()
        mean, scale = a['train_mean'].copy(), a['train_std'].copy()
    test_raw = raw[cfg['validation_end']:cfg['test_end']].copy()
    epsilon = np.maximum(1e-8, 1e-6*np.mean(abs(raw[:cfg['train_end']]), axis=0))
    local_origins = origins-cfg['validation_end']
    if local_origins.min() < 0 or local_origins.max()+cfg['pred_len'] > len(test_raw):
        raise ValueError('Invalid held-out target boundaries')
    hashes = {str((source/'models'/f'{r["variant"]}_{r["seed"]}'/'test_pred.npy').resolve()):
              hashlib.sha256((source/'models'/f'{r["variant"]}_{r["seed"]}'/'test_pred.npy').read_bytes()).hexdigest()
              for r in records}
    protocol = dict(source_experiment=str(source), config=cfg, test_windows=len(origins),
        reference_block=[cfg['validation_end'], cfg['test_end']],
        references=['raw', 'drop_imf1', 'drop_imf1_imf2'], columns=previous_protocol['columns'],
        forecast_files_sha256=hashes, original_checkpoints_and_forecasts_unchanged=True,
        retrained=False, reselected=False,
        target_processing='Offline CEEMDAN on the held-out test block only; no filtered input is supplied to models',
        interpretation='Retrospective smooth-target accuracy, not improvement in raw-series forecasting accuracy',
        mape='100*mean(|forecast-target|/|target|), excluding |target|<=epsilon per channel',
        wmape='100*sum(|forecast-target|)/sum(|target|), including zero-target errors',
        main_aggregation='Equal-weight macro mean across 7 separately computed variable metrics',
        pooled_wmape='Also supplied; pools original units, implicitly upweights large-magnitude variables',
        epsilon_original_units=epsilon.tolist(), epsilon_rule='max(1e-8, 1e-6*training mean absolute value)',
        percentages_computed_after_inverse_scaling=True,
        comparison_rule='All three training methods share an identical target reference within each comparison table')
    atomic_json(out/'protocol.json', protocol)
    atomic_json(out/'status.json', dict(status='decomposing_test_references'))
    clean1, clean2 = np.empty_like(test_raw), np.empty_like(test_raw)
    first_all, second_all = np.empty_like(test_raw), np.empty_like(test_raw)
    diagnostics = []
    tasks = [(c, test_raw[:, c], cfg) for c in range(test_raw.shape[1])]
    with mp.get_context('spawn').Pool(cfg['workers']) as pool:
        for c, a, b, first, second, info in pool.imap_unordered(denoise_channel, tasks):
            clean1[:, c], clean2[:, c] = a, b
            first_all[:, c], second_all[:, c] = first, second
            diagnostics.append(info)
    references = dict(raw=test_raw, drop_imf1=clean1, drop_imf1_imf2=clean2)
    np.testing.assert_allclose(clean1+first_all, test_raw, atol=1e-10)
    np.testing.assert_allclose(clean2+first_all+second_all, test_raw, atol=1e-10)
    np.savez_compressed(out/'test_reference_series.npz', **references, imf1=first_all, imf2=second_all,
                        original_unit_epsilon=epsilon, window_origins=origins)
    atomic_json(out/'imf_diagnostics.json', sorted(diagnostics,key=lambda x:x['channel']))
    physical_targets = {key: np.stack([values[t:t+cfg['pred_len']] for t in local_origins])
                        for key, values in references.items()}
    scaled_targets = {key: ((values-mean)/scale).astype(np.float32) for key, values in physical_targets.items()}
    np.testing.assert_array_equal(scaled_targets['raw'],np.load(source/'test_true.npy'))
    measured = []
    for record in records:
        pred_path = source/'models'/f'{record["variant"]}_{record["seed"]}'/'test_pred.npy'
        pred_scaled = np.load(pred_path)
        pred_physical = pred_scaled.astype(np.float64)*scale+mean
        for reference in VARIANTS:
            result = scores(pred_scaled,scaled_targets[reference],pred_physical,
                            physical_targets[reference],epsilon)
            if reference == 'raw':
                np.testing.assert_allclose([result['mse'],result['mae']],
                    [record['test']['mse'],record['test']['mae']],rtol=1e-12,atol=1e-12)
            measured.append(dict(training_label=record['variant'],reference_label=reference,
                                 seed=record['seed'],metrics=result))
    atomic_json(out/'results.json',measured)
    summary = {}
    for reference in VARIANTS:
        summary[reference] = {}
        for variant in VARIANTS:
            items = [r['metrics'] for r in measured if r['training_label']==variant and r['reference_label']==reference]
            summary[reference][variant] = {}
            for metric in ['mse','mae','mape_macro_percent','wmape_macro_percent','wmape_pooled_percent']:
                values = np.array([r[metric] for r in items],dtype=float)
                summary[reference][variant][metric] = dict(mean=float(values.mean()),std=float(values.std(ddof=1)),seeds=values.tolist())
    atomic_json(out/'summary.json',summary)
    names = dict(raw='原始标签',drop_imf1='删除 IMF1',drop_imf1_imf2='删除 IMF1＋IMF2')
    sections = []
    for reference in VARIANTS:
        lines = []
        for variant in VARIANTS:
            s = summary[reference][variant]
            lines.append(f"| {names[variant]} | {s['mse']['mean']:.6f} | {s['mae']['mean']:.6f} | {s['mape_macro_percent']['mean']:.2f}% | {s['wmape_macro_percent']['mean']:.2f}% | {s['wmape_pooled_percent']['mean']:.2f}% |")
        sections.append(f'## 评价对照：{names[reference]}\n\n| 训练标签 | MSE | MAE | MAPE（变量平均） | WMAPE（变量平均） | WMAPE（总体合并） |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(lines))
    coverage = []
    for reference in VARIANTS:
        item = next(r['metrics'] for r in measured if r['reference_label']==reference)
        for column, threshold, excluded in zip(protocol['columns'],epsilon,item['mape_excluded_per_channel']):
            coverage.append(f'| {names[reference]} | {column} | {threshold:.8g} | {excluded} / {item["observations_per_channel"]} |')
    report = ('# 用去噪后的新标签重新评价已有 PatchTST 模型\n\n'
        '复用上次 3 种训练标签 × 3 个种子的 9 份固定预测，每种方法均使用完整 2,785 个测试窗口、7 个变量、96 步。'
        '本次未训练模型、未重新选择检查点。分别以同一原始测试序列、删除 IMF1、删除 IMF1+IMF2 为共同对照，避免不同方法各自对照不同目标。\n\n'+
        '\n\n'.join(sections)+'\n\n## 指标口径\n\n'
        '- MSE/MAE 使用同一原始训练段统计量标准化，便于合并多变量。\n'
        '- MAPE 和 WMAPE 在还原原始单位后计算；MAPE 分母取目标绝对值。\n'
        '- MAPE = 100 × mean(|预测−目标| / |目标|)，只排除绝对值不大于固定 epsilon 的目标，未对其余分母截断。'
        'epsilon 仅由原训练段确定；原始/去噪目标内的所有模型共享同一有效点掩码。\n'
        '- WMAPE = 100 × sum(|预测−目标|) / sum(|目标|)，零目标处的误差保留在分子。\n'
        '- 主报告先逐变量计算 MAPE/WMAPE，再对 7 个变量等权平均，避免负载与温度单位直接混合。'
        '另给总体合并 WMAPE，后者按原始量级隐式加权，不能等同变量平均。\n'
        '- 指标先分别对各随机种子计算，再取均值；不是先平均预测。详细结果含每变量指标及种子标准差。\n\n'
        '## 评价目标与边界\n\n'
        '新测试标签是对固定测试时段离线 CEEMDAN 后重构得到，参数与训练标签预处理相同（32 trials、固定种子、先提取两个 IMF）。'
        '这一参考可能利用测试段内较晚观测，但它只用于事后评价，未作为输入、监督或选模依据。'
        '因此它衡量“预测平滑信号”的能力，不能解释成对原始观测的预测精度提升。'
        '检查点仍是此前按原始验证标签选择的版本，并非按新目标重新选模。\n\n'
        '## MAPE 的近零排除量（按窗口内预测元素计数）\n\n'
        '| 评价目标 | 变量 | 原始单位 epsilon | 排除数 / 总数 |\n|---|---|---:|---:|\n'+
        '\n'.join(coverage)+'\n')
    (out/'REPORT.zh.md').write_text(report)
    for path, expected_hash in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_hash
    atomic_json(out/'verification.json',dict(status='passed',fixed_predictions_unchanged=True,
        original_reference_metrics_reproduced=True,new_target_reconstruction_verified=True,
        percentages_in_original_units=True,comparisons_share_target_and_mask=True,
        models=9,reference_types=3))
    atomic_json(out/'status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start))
    print(report,flush=True)


if __name__ == '__main__':
    main()
