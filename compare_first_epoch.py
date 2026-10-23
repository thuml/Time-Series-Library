"""Compare the existing fusion epoch with freshly trained original baselines.

All models use the full official validation split, not a sampled test metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from exp.exp_main import Exp_Main
from quick_validate import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion-checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.fusion_checkpoint_dir / 'config.json').read_text())
    hist = json.loads((args.fusion_checkpoint_dir / 'history.json').read_text())
    # Freeze the currently saved best epoch and select the corresponding record.
    best = min(hist, key=lambda row: row['val_loss'])
    if best['epoch'] != 1:
        raise ValueError('Current checkpoint is no longer epoch 1; do not mislabel later weights')
    frozen = out / 'fusion_epoch1.pth'
    if not frozen.exists():
        shutil.copy2(args.fusion_checkpoint_dir / 'checkpoint.pth', frozen)
    atomic_json(out / 'fusion_config.json', cfg)
    atomic_json(out / 'fusion_history.json', hist)
    protocol = dict(dataset=cfg['data'], features=cfg['features'], seq_len=cfg['seq_len'],
                    pred_len=cfg['pred_len'], validation_windows=2785,
                    training_batches=2112, executed_epochs=1,
                    scheduler_epochs=cfg['train_epochs'], random_seed=cfg['random_seed'],
                    fusion_checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                    note='Full validation MSE; same-width first-epoch comparison, not tuned final accuracy',
                    test_split_used=False)
    atomic_json(out / 'protocol.json', protocol)
    results = {'CEEMDANPatchTST': dict(epoch=1, train_mse=best['train_loss'],
                                    validation_mse=best['val_loss'],
                                    train_and_validation_seconds=best['seconds'])}
    atomic_json(out / 'results.json', results)
    torch.set_num_threads(1)
    start = time.time()
    for name in ['PatchTST', 'Autoformer']:
        random.seed(cfg['random_seed'])
        np.random.seed(cfg['random_seed'])
        torch.manual_seed(cfg['random_seed'])
        conf = dict(cfg, model=name, stop_after_epochs=1, checkpoint_interval=100,
                    max_train_steps=0, max_eval_steps=0, use_gpu=False,
                    use_multi_gpu=False, use_amp=False, num_workers=0,
                    checkpoints=str(out / 'baseline_checkpoints'))
        atomic_json(out / 'status.json', dict(status='running', model=name,
                    elapsed_seconds=time.time() - start))
        exp = Exp_Main(SimpleNamespace(**conf))
        exp.train(f'{name}_first_epoch')
        folder = out / 'baseline_checkpoints' / f'{name}_first_epoch'
        record = json.loads((folder / 'history.json').read_text())[0]
        results[name] = dict(epoch=1, train_mse=record['train_loss'],
                             validation_mse=record['val_loss'],
                             train_and_validation_seconds=record['seconds'])
        atomic_json(out / 'results.json', results)
    rows = []
    fusion_mse = results['CEEMDANPatchTST']['validation_mse']
    for name in ['CEEMDANPatchTST', 'PatchTST', 'Autoformer']:
        record = results[name]
        rows.append(f"| {name} | {record['train_mse']:.6f} | {record['validation_mse']:.6f} |")
    notes = []
    for name in ['PatchTST', 'Autoformer']:
        change = (fusion_mse / results[name]['validation_mse'] - 1) * 100
        notes.append(f"融合模型相对 {name} 的验证 MSE 变化为 {change:+.2f}%（负数更好）。")
    report = ('# 完整配置首轮对比\n\nETTh1、7 变量、96→96；每个模型训练完整一轮，'
              '使用全部 2,785 个验证窗口，未使用测试集。优化器、学习率日程、'
              '批大小和种子相同。调度仍按原 30 轮设置，只提前在一轮后停止基线。\n\n'
              '| 模型 | 首轮训练 MSE | 完整验证 MSE |\n|---|---:|---:|\n' +
              '\n'.join(rows) + '\n\n' + '\n\n'.join(notes) +
              '\n\n这是固定宽度、单种子、首轮预算的阶段性比较。两个原模型未分别调优或训练至收敛，'
              '不能作为论文最佳结果比较。各模型完整经历相同训练窗口数，'
              '但不同初始化与 dropout 会影响训练窗口顺序，未声称逐批顺序相同。'
              '融合权重已单独冻结，原长训练继续运行。\n')
    (out / 'REPORT.zh.md').write_text(report)
    atomic_json(out / 'status.json', dict(status='complete', elapsed_seconds=time.time() - start))
    print(report, flush=True)


if __name__ == '__main__':
    main()
