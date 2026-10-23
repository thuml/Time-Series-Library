"""Bounded, paired validation pilot. Never reads the test split.

This is a new short training run, not a snapshot of an uninstrumented running
Python process. Architecture is loaded from the full run's saved config.
"""
import argparse
from datetime import datetime
from pathlib import Path
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from data_provider.data_loader import Dataset_ETT_hour
from models.CEEMDANPatchTST import Model as Fusion
from models.PatchTST import Model as PatchTST


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def checkpoint(path, model, optimizer, scheduler, step, cfg, train_indices):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), step=step, config=cfg,
                    train_indices=train_indices.tolist(),
                    torch_rng=torch.get_rng_state()), tmp)
    os.replace(tmp, path)


def batch(dataset, indices, horizon):
    pairs = [dataset[int(i)] for i in indices]
    x = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    y = torch.tensor(np.stack([p[1][-horizon:] for p in pairs]), dtype=torch.float32)
    return x, y


def evaluate(model, dataset, indices, cfg):
    predictions, labels = [], []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for begin in range(0, len(indices), cfg['batch_size']):
            x, y = batch(dataset, indices[begin:begin + cfg['batch_size']], cfg['pred_len'])
            prediction = (x[:, -1:].expand(-1, cfg['pred_len'], -1)
                          if model is None else model(x))
            predictions.append(prediction.numpy())
            labels.append(y.numpy())
    pred, true = np.concatenate(predictions), np.concatenate(labels)
    error = pred.astype(np.float64) - true.astype(np.float64)
    metrics = dict(mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
                   windows=len(indices), per_channel_mse=np.mean(error**2, axis=(0, 1)).tolist())
    if model is not None:
        model.train()
    return metrics, pred, true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--validation-windows', type=int, default=32)
    args = parser.parse_args()
    if args.steps < 16 or args.validation_windows < 16:
        parser.error('Require >=16 training steps and >=16 validation windows')
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'config.json', dict(source_config=str(args.config),
                full_model_config=cfg, train_steps=args.steps,
                validation_windows=args.validation_windows,
                protocol='new short run; same architecture and Adam/MSE/OneCycle; shortened schedule',
                test_split_used=False))
    load = lambda flag: Dataset_ETT_hour(
        root_path=cfg['root_path'], data_path=cfg['data_path'], flag=flag,
        size=[cfg['seq_len'], cfg['label_len'], cfg['pred_len']],
        features=cfg['features'], target=cfg['target'], timeenc=1, freq=cfg['freq'])
    train, val = load('train'), load('val')
    rng = np.random.default_rng(cfg['random_seed'])
    train_indices = rng.choice(len(train), args.steps * cfg['batch_size'], replace=False)
    val_indices = np.linspace(0, len(val) - 1, args.validation_windows, dtype=int)
    # Early validation samples cover the same full span as the final subset.
    early_indices = val_indices[np.linspace(0, len(val_indices) - 1, 16, dtype=int)]
    np.savez(output / 'sample_indices.npz', train=train_indices, validation=val_indices,
             early_validation=early_indices)
    results = {'protocol': 'validation only; preliminary, one seed, limited training',
               'full_training_process_unchanged': True, 'models': {}}
    for label, indices in [('early', early_indices), ('final', val_indices)]:
        metrics, pred, true = evaluate(None, val, indices, cfg)
        results.setdefault('persistence', {})[label] = metrics
        np.savez(output / f'persistence_{label}.npz', pred=pred, true=true, indices=indices)
    atomic_json(output / 'results.json', results)
    started = time.time()
    for name, constructor in [('PatchTST', PatchTST), ('CEEMDANPatchTST', Fusion)]:
        random.seed(cfg['random_seed'])
        np.random.seed(cfg['random_seed'])
        torch.manual_seed(cfg['random_seed'])
        model = constructor(SimpleNamespace(**cfg)).float().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg['learning_rate'], total_steps=args.steps,
            pct_start=cfg['pct_start'])
        losses = []
        results['models'][name] = {}
        for step in range(1, args.steps + 1):
            selected = train_indices[(step - 1) * cfg['batch_size']:step * cfg['batch_size']]
            x, y = batch(train, selected, cfg['pred_len'])
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {name} at step {step}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 8 == 0 or step == args.steps:
                checkpoint(output / f'{name}_latest.pt', model, optimizer, scheduler,
                           step, cfg, train_indices)
                atomic_json(output / 'status.json', dict(status='running', model=name,
                            step=step, total_steps=args.steps, train_mse_mean=float(np.mean(losses)),
                            elapsed_seconds=time.time() - started,
                            updated_at=datetime.now().astimezone().isoformat()))
                print(f'{name}: step {step}/{args.steps}, train MSE mean={np.mean(losses):.6f}', flush=True)
            if step in (16, args.steps):
                label = 'early' if step == 16 and step != args.steps else 'final'
                indices = early_indices if label == 'early' else val_indices
                metrics, pred, true = evaluate(model, val, indices, cfg)
                metrics['train_steps'] = step
                metrics['train_windows_seen'] = step * cfg['batch_size']
                results['models'][name][label] = metrics
                np.savez(output / f'{name}_{label}.npz', pred=pred, true=true, indices=indices)
                checkpoint(output / f'{name}_step{step}.pt', model, optimizer,
                           scheduler, step, cfg, train_indices)
                atomic_json(output / 'results.json', results)
                print(f'{name}: {label} validation {json.dumps(metrics)}', flush=True)
    rows = []
    for name in ['CEEMDANPatchTST', 'PatchTST']:
        m = results['models'][name]['final']
        rows.append(f"| {name} | {m['mse']:.6f} | {m['mae']:.6f} |")
    m = results['persistence']['final']
    rows.append(f"| 延续预测 | {m['mse']:.6f} | {m['mae']:.6f} |")
    text = ('# 快速验证结果\n\n这是独立的短训练，不是原有长训练的中途权重。'
            f"每个可训练模型训练 {args.steps} 批、{args.steps * cfg['batch_size']} 个窗口，"
            f"在均匀覆盖验证区间的 {args.validation_windows} 个窗口上比较；未使用测试集。\n\n"
            '| 模型 | 标准化 MSE | 标准化 MAE |\n|---|---:|---:|\n' + '\n'.join(rows) +
            '\n\n单种子、少量训练与验证样本只能用于早期筛查，不能证明结构稳定优于基线。'
            '学习率沿用 OneCycle 机制，但周期缩短到本次预算，不能与原长训练同一步数直接等同。\n')
    (output / 'REPORT.zh.md').write_text(text)
    atomic_json(output / 'status.json', dict(status='complete', elapsed_seconds=time.time() - started))
    print(text, flush=True)


if __name__ == '__main__':
    main()
