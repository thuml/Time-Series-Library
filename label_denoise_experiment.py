"""Paired PatchTST experiment: raw inputs, CEEMDAN-denoised training labels.

Only the training time segment is decomposed. Validation/test targets remain
raw. IMF1 and IMF2 are candidate high-frequency noise, not assumed pure noise.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN, EMD
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.PatchTST import Model
from quick_validate import atomic_json

VARIANTS = ['raw', 'drop_imf1', 'drop_imf1_imf2']
RUN_OUTPUT = None


def signal_diagnostics(x):
    centered = x - x.mean()
    power = abs(np.fft.rfft(centered))**2
    freq = np.fft.rfftfreq(len(x))
    active = power[1:]
    floor = max(float(active.max()) * 1e-15, 1e-30)
    return dict(spectral_centroid=float(np.sum(freq * power) / max(float(power.sum()), 1e-30)),
                spectral_flatness=float(np.exp(np.mean(np.log(active + floor))) / np.mean(active + floor)),
                lag1_correlation=float(np.sum(centered[:-1] * centered[1:]) /
                                       max(float(np.sum(centered**2)), 1e-30)))


def denoise_channel(task):
    channel, x, cfg = task
    torch.set_num_threads(1)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected one finite training channel')
    if np.std(x) < 1e-10:
        modes = np.stack([np.zeros_like(x), np.zeros_like(x), x])
        actual_imfs = 0
    else:
        ce = CEEMDAN(trials=cfg['ceemdan_trials'], epsilon=cfg['ceemdan_epsilon'],
                     ext_EMD=EMD(MAX_ITERATION=cfg['ceemdan_max_sift']), parallel=False)
        ce.noise_seed(cfg['ceemdan_seed'])
        modes = ce.ceemdan(x, max_imf=2)
        actual_imfs = len(modes) - 1
        if actual_imfs < 2:
            raise RuntimeError(f'Channel {channel}: fewer than two IMFs; cannot silently drop remainder')
    # Truncating after two IMFs leaves all remaining structure in the remainder.
    # No need to extract later IMFs merely to add them back together.
    first, second = modes[0], modes[1]
    np.testing.assert_allclose(modes.sum(0), x, atol=1e-8, rtol=1e-8)
    clean1, clean2 = x - first, x - first - second
    info = dict(channel=channel, actual_extracted_imfs=actual_imfs,
                first=signal_diagnostics(first), second=signal_diagnostics(second),
                first_variance_ratio=float(np.var(first)/max(float(np.var(x)), 1e-30)),
                second_variance_ratio=float(np.var(second)/max(float(np.var(x)), 1e-30)))
    return channel, clean1, clean2, first, second, info


class Windows(Dataset):
    def __init__(self, raw_inputs, labels, origins, seq_len, pred_len):
        self.x, self.y = raw_inputs, labels
        self.origins = np.asarray(origins, dtype=int)
        self.seq_len, self.pred_len = seq_len, pred_len
        if len(self.origins) == 0 or self.origins.min() < seq_len:
            raise ValueError('Invalid history origins')
        if self.origins.max() + pred_len > min(len(raw_inputs), len(labels)):
            raise ValueError('A label window crosses the available time boundary')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        t = int(self.origins[index])
        return (self.x[t-self.seq_len:t], self.y[t:t+self.pred_len],
                self.x[t:t+self.pred_len])


def model_config(cfg, channels):
    return SimpleNamespace(enc_in=channels, seq_len=cfg['seq_len'], pred_len=cfg['pred_len'],
        e_layers=cfg['e_layers'], n_heads=cfg['n_heads'], d_model=cfg['d_model'],
        d_ff=cfg['d_ff'], dropout=cfg['dropout'], fc_dropout=0., head_dropout=0.,
        individual=False, patch_len=cfg['patch_len'], stride=cfg['stride'], padding_patch='end',
        revin=True, affine=True, subtract_last=False, decomposition=False, kernel_size=25)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, model, optimizer, scheduler, **metadata):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), **metadata), tmp)
    os.replace(tmp, path)


def eval_predictions(model, loader):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, _, raw_y in loader:
            pred.append(model(x.float()).numpy())
            true.append(raw_y.numpy())
    p, y = np.concatenate(pred), np.concatenate(true)
    if not np.isfinite(p).all():
        raise RuntimeError('Nonfinite forecast')
    return p, y


def error_metrics(pred, true, scale):
    e = pred.astype(np.float64) - true.astype(np.float64)
    physical = e * scale[None, None, :]
    by_horizon = {}
    for h in [1, 2, 5, 10, 96]:
        if h <= e.shape[1]:
            z = e[:, :h]
            by_horizon[str(h)] = dict(mse=float(np.mean(z**2)), mae=float(np.mean(abs(z))))
    return dict(mse=float(np.mean(e**2)), mae=float(np.mean(abs(e))),
        per_channel_mse=np.mean(e**2, axis=(0, 1)).tolist(),
        per_channel_mae=np.mean(abs(e), axis=(0, 1)).tolist(),
        original_units_per_channel_rmse=np.sqrt(np.mean(physical**2, axis=(0, 1))).tolist(),
        original_units_per_channel_mae=np.mean(abs(physical), axis=(0, 1)).tolist(),
        prefix_horizons=by_horizon, windows=len(pred))


def train_one(cfg, variant, seed, inputs, labels, origins, val_loader, scale, out, deadline):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Model(model_config(cfg, inputs.shape[1])).float()
    initialization = state_hash(model)
    data = Windows(inputs, labels, origins, cfg['seq_len'], cfg['pred_len'])
    generator = torch.Generator().manual_seed(seed + 10000)
    loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=cfg['learning_rate'], epochs=cfg['max_epochs'], steps_per_epoch=len(loader),
        pct_start=cfg['pct_start'])
    folder = out / 'models' / f'{variant}_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    best, stale, history = math.inf, 0, []
    started = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        if time.perf_counter() > deadline:
            raise TimeoutError('Experiment time budget reached; saved results remain available')
        model.train()
        label_sum, raw_sum, elements = 0., 0., 0
        for x, target, raw_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.float())
            loss = nn.functional.mse_loss(prediction, target.float())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
            label_sum += float(loss.detach()) * target.numel()
            raw_sum += float(nn.functional.mse_loss(prediction.detach(), raw_y)) * target.numel()
            elements += target.numel()
        pred, true = eval_predictions(model, val_loader)
        val = error_metrics(pred, true, scale)
        row = dict(epoch=epoch, training_label_mse=label_sum/elements,
                   training_raw_mse=raw_sum/elements, validation_raw_mse=val['mse'],
                   elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        if val['mse'] < best:
            best, stale = val['mse'], 0
            best_epoch = epoch
            save_checkpoint(folder / 'best.pt', model, optimizer, scheduler,
                            variant=variant, seed=seed, epoch=epoch, config=cfg,
                            validation_raw_metrics=val, initial_state_sha256=initialization)
        else:
            stale += 1
        save_checkpoint(folder / 'latest.pt', model, optimizer, scheduler,
                        variant=variant, seed=seed, epoch=epoch, config=cfg)
        atomic_json(folder / 'history.json', history)
        atomic_json(out / 'status.json', dict(status='training', variant=variant, seed=seed,
                    epoch=epoch, max_epochs=cfg['max_epochs'], best_validation_mse=best))
        if epoch % 5 == 0 or stale >= cfg['patience']:
            print(f'TRAIN {variant} seed={seed} epoch={epoch}: raw val MSE={val["mse"]:.6f}, best={best:.6f}', flush=True)
        if stale >= cfg['patience']:
            break
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    pred, true = eval_predictions(model, val_loader)
    val_metrics = error_metrics(pred, true, scale)
    assert abs(val_metrics['mse'] - best) < 1e-8
    np.save(folder / 'validation_pred.npy', pred)
    return dict(variant=variant, seed=seed, best_epoch=best_epoch, epochs_run=len(history),
                validation=val_metrics, seconds=time.perf_counter()-started,
                initial_state_sha256=initialization)


def aggregate(records, split):
    result = {}
    for variant in VARIANTS:
        selected = [r for r in records if r['variant'] == variant]
        result[variant] = {}
        for metric in ['mse', 'mae']:
            values = np.array([r[split][metric] for r in selected])
            result[variant][metric] = dict(mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values)>1 else 0., seeds=values.tolist())
    return result


def main():
    global RUN_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/label_denoise_pilot.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT = out
    start = time.perf_counter()
    deadline = start + cfg['max_seconds']
    torch.set_num_threads(1)
    df = pd.read_csv(cfg['data_path'])
    columns = df.columns[1:].tolist()
    raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if cfg['test_end'] > len(raw) or not np.isfinite(raw).all():
        raise ValueError('Invalid time-series data')
    train_raw = raw[:cfg['train_end']].copy()
    mean, scale = train_raw.mean(0), train_raw.std(0)
    scale[scale < 1e-10] = 1.
    raw_scaled = ((raw - mean)/scale).astype(np.float32)
    rng = np.random.default_rng(cfg['sampling_seed'])
    possible = np.arange(cfg['seq_len'], cfg['train_end']-cfg['pred_len']+1)
    origins = np.sort(rng.choice(possible, cfg['train_windows'], replace=False))
    all_val = np.arange(cfg['train_end'], cfg['validation_end']-cfg['pred_len']+1)
    val_origins = all_val[np.linspace(0, len(all_val)-1, cfg['validation_windows'], dtype=int)]
    test_origins = np.arange(cfg['validation_end'], cfg['test_end']-cfg['pred_len']+1)
    np.savez(out / 'sample_origins.npz', train=origins, validation=val_origins, test=test_origins,
             train_mean=mean, train_std=scale)
    protocol = dict(config=cfg, columns=columns,
        data_sha256=hashlib.sha256(Path(cfg['data_path']).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_target_interval=[0, cfg['train_end']],
        label_preprocessing='Offline CEEMDAN on the TRAIN segment only, independently per channel',
        candidates='Drop IMF1; drop IMF1+IMF2. High-frequency candidates are not proven pure noise.',
        input='Identical raw histories in every condition',
        validation_and_test_targets='Original raw data normalized using raw training statistics only',
        selection='Best epoch per run and denoising choice use original validation targets only',
        test_windows=len(test_origins), paired_initializations=True, paired_sample_orders=True,
        label_offline_noncausal_within_training_segment=True,
        excluded_future='No validation/test data used for decomposition, scaling, or training labels')
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'status.json', dict(status='decomposing', training_hours=len(train_raw)))
    clean1, clean2 = np.empty_like(train_raw), np.empty_like(train_raw)
    imf1, imf2 = np.empty_like(train_raw), np.empty_like(train_raw)
    diagnostics = []
    tasks = [(c, train_raw[:, c].copy(), cfg) for c in range(len(columns))]
    with mp.get_context('spawn').Pool(cfg['workers']) as pool:
        for c, a, b, first, second, info in pool.imap_unordered(denoise_channel, tasks):
            clean1[:, c], clean2[:, c], imf1[:, c], imf2[:, c] = a, b, first, second
            diagnostics.append(info)
            print(f'CEEMDAN {columns[c]} complete, elapsed {time.perf_counter()-start:.1f}s', flush=True)
    preprocessing_seconds = time.perf_counter()-start
    np.savez_compressed(out / 'training_labels.npz', raw=train_raw, drop_imf1=clean1,
                       drop_imf1_imf2=clean2, imf1=imf1, imf2=imf2)
    atomic_json(out / 'imf_diagnostics.json', sorted(diagnostics, key=lambda a: a['channel']))
    labels = dict(raw=raw_scaled[:cfg['train_end']],
                  drop_imf1=((clean1-mean)/scale).astype(np.float32),
                  drop_imf1_imf2=((clean2-mean)/scale).astype(np.float32))
    val_data = Windows(raw_scaled, raw_scaled, val_origins, cfg['seq_len'], cfg['pred_len'])
    val_loader = DataLoader(val_data, batch_size=cfg['batch_size'], num_workers=0)
    # Benchmark before fitting; initialization is reset for every real run.
    bench = Model(model_config(cfg, len(columns))).train()
    x = torch.from_numpy(np.stack([raw_scaled[t-cfg['seq_len']:t] for t in origins[:cfg['batch_size']]]))
    y = torch.from_numpy(np.stack([raw_scaled[t:t+cfg['pred_len']] for t in origins[:cfg['batch_size']]]))
    optimizer = torch.optim.Adam(bench.parameters(), lr=cfg['learning_rate'])
    benchmark_start = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(bench(x), y).backward()
        optimizer.step()
    step_seconds = (time.perf_counter()-benchmark_start)/10
    del bench, optimizer
    estimate = step_seconds * math.ceil(len(origins)/cfg['batch_size']) * cfg['max_epochs'] * len(cfg['seeds']) * len(VARIANTS) * 1.6
    if estimate + preprocessing_seconds > cfg['max_seconds']-120:
        raise RuntimeError(f'Estimated budget {estimate:.0f}s too large; reduce protocol before training, not after test evaluation')
    atomic_json(out / 'timing_plan.json', dict(preprocessing_seconds=preprocessing_seconds,
                benchmark_step_seconds=step_seconds, estimated_training_and_evaluation_seconds=estimate))
    print(f'BENCHMARK step={step_seconds:.4f}s; estimated training budget={estimate:.0f}s', flush=True)
    records = []
    for seed in cfg['seeds']:
        hashes = []
        for variant in VARIANTS:
            record = train_one(cfg, variant, seed, raw_scaled, labels[variant], origins,
                               val_loader, scale, out, deadline)
            hashes.append(record['initial_state_sha256'])
            records.append(record)
            atomic_json(out / 'training_results.json', records)
        assert len(set(hashes)) == 1, 'Paired initializations differ'
    val_summary = aggregate(records, 'validation')
    selected = min(VARIANTS[1:], key=lambda v: val_summary[v]['mse']['mean'])
    atomic_json(out / 'selection_before_test.json', dict(selected_denoising=selected,
                criterion='Mean original-validation MSE across seeds', validation=val_summary))
    test_data = Windows(raw_scaled, raw_scaled, test_origins, cfg['seq_len'], cfg['pred_len'])
    test_loader = DataLoader(test_data, batch_size=cfg['batch_size'], num_workers=0)
    for record in records:
        if time.perf_counter() > deadline:
            raise TimeoutError('Time budget exceeded before final test evaluation')
        folder = out / 'models' / f'{record["variant"]}_{record["seed"]}'
        saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        model = Model(model_config(cfg, len(columns))).eval()
        model.load_state_dict(saved['model'])
        pred, true = eval_predictions(model, test_loader)
        np.save(folder / 'test_pred.npy', pred)
        if not (out / 'test_true.npy').exists():
            np.save(out / 'test_true.npy', true)
        record['test'] = error_metrics(pred, true, scale)
        atomic_json(out / 'results.json', records)
    test_summary = aggregate(records, 'test')
    atomic_json(out / 'summary.json', dict(validation=val_summary, test=test_summary,
                selected_denoising=selected, preprocessing_seconds=preprocessing_seconds,
                total_seconds=time.perf_counter()-start))
    write_report(out, cfg, records, val_summary, test_summary, selected, columns,
                 preprocessing_seconds, time.perf_counter()-start)
    atomic_json(out / 'status.json', dict(status='complete', elapsed_seconds=time.perf_counter()-start))
    print((out/'REPORT.zh.md').read_text(), flush=True)


def write_report(out, cfg, records, val, test, selected, columns, preprocessing, elapsed):
    names = dict(raw='原始 label', drop_imf1='删除 IMF1', drop_imf1_imf2='删除 IMF1＋IMF2')
    rows = []
    for v in VARIANTS:
        change = (test[v]['mse']['mean']/test['raw']['mse']['mean']-1)*100
        rows.append(f"| {names[v]} | {val[v]['mse']['mean']:.6f} | {test[v]['mse']['mean']:.6f} ± {test[v]['mse']['std']:.6f} | {test[v]['mae']['mean']:.6f} | {change:+.2f}% |")
    seed_rows = []
    for seed in cfg['seeds']:
        values = {r['variant']: r['test']['mse'] for r in records if r['seed']==seed}
        seed_rows.append(f"| {seed} | {values['raw']:.6f} | {values['drop_imf1']:.6f} | {values['drop_imf1_imf2']:.6f} |")
    horizon_rows = []
    for h in ['1','2','5','10','96']:
        means = {v: np.mean([r['test']['prefix_horizons'][h]['mse'] for r in records if r['variant']==v]) for v in VARIANTS}
        horizon_rows.append(f"| {h} | {means['raw']:.6f} | {means['drop_imf1']:.6f} | {means['drop_imf1_imf2']:.6f} |")
    report = (f'# CEEMDAN 仅去噪训练标签：PatchTST 对照实验\n\n'
        f'ETTh1，7 变量，96→96。训练窗口 {cfg["train_windows"]} 个，验证窗口 {cfg["validation_windows"]} 个，'
        f'测试使用完整 {records[0]["test"]["windows"]} 个窗口；{len(cfg["seeds"])} 个随机种子。'
        '输入始终是原始历史序列，只有训练 label 改变。验证选模及最终测试均对照原始序列。\n\n'
        '| 训练 label | 验证 MSE 均值 | 测试 MSE 均值 ± 种子标准差 | 测试 MAE 均值 | 测试 MSE 相对原始 label |\n'
        '|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+
        '\n\n误差为原始目标经同一训练段均值/标准差标准化后的误差，不是去噪目标上的误差。'
        f'验证集预选的去噪方案是 **{names[selected]}**，选择在测试评估前落盘。\n\n'
        '## 各种子测试 MSE\n\n| 种子 | 原始 label | 删除 IMF1 | 删除 IMF1＋IMF2 |\n|---|---:|---:|---:|\n'+
        '\n'.join(seed_rows)+'\n\n## 前 h 步平均测试 MSE（三种子均值）\n\n'
        '| h | 原始 label | 删除 IMF1 | 删除 IMF1＋IMF2 |\n|---|---:|---:|---:|\n'+
        '\n'.join(horizon_rows)+'\n\n## 方法和边界\n\n'
        '- CEEMDAN 只分解前 8,640 个小时的训练段，每个变量独立、32 次噪声试验，提取前两个 IMF 后停止；余项保留所有更慢结构。'
        '先删除 IMF，再将其余 IMF 与余项相加，等价于原序列减去被删 IMF。\n'
        '- IMF1/IMF2 是预先固定的高频疑似噪声候选，不保证纯噪声。诊断见 imf_diagnostics.json；本试验没有按测试结果挑选分量。\n'
        '- 训练标签为离线处理，可以利用同一训练段内较晚的观测；不使用验证/测试数据。它不是在线因果滤波器。\n'
        '- 三组使用相同模型、原始输入、采样起点、初始权重、每轮样本顺序、Adam、OneCycle 预算。最佳 epoch 与早停均依据原始验证标签。\n'
        '- 所有种子/条件训练并选模完成后，才评估测试集；每次检查点和完整预测均保留。\n'
        '- 这是小样本、单数据集研究试验。种子标准差不是时间相关样本的置信区间，不能凭单次小幅提升宣称稳定有效。\n\n'
        f'预处理 {preprocessing:.1f} 秒，总耗时 {elapsed:.1f} 秒（{elapsed/60:.1f} 分钟）。'
        '配置与来源指纹见 protocol.json；详细指标含每变量原始单位误差，见 results.json。\n')
    (out / 'REPORT.zh.md').write_text(report)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if RUN_OUTPUT is not None:
            atomic_json(RUN_OUTPUT / 'status.json', dict(status='failed', error=repr(error)))
        raise
