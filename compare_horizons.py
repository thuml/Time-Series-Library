"""Paired short-horizon evaluation of frozen 96-output first-epoch models.

Pool workers each load the model once; independent fixed-seed CEEMDAN calls
and evaluation-mode normalization preserve serial predictions. No training.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.data_loader import Dataset_ETT_hour
from models.CEEMDANPatchTST import Model as Fusion
from models.PatchTST import Model as PatchTST
from quick_validate import atomic_json


_model = None
_dataset = None


def dataset(cfg):
    return Dataset_ETT_hour(root_path=cfg['root_path'], data_path=cfg['data_path'],
        flag='val', size=[cfg['seq_len'], cfg['label_len'], cfg['pred_len']],
        features=cfg['features'], target=cfg['target'], timeenc=1, freq=cfg['freq'])


def init_worker(cfg, weight):
    global _model, _dataset
    torch.set_num_threads(1)
    _dataset = dataset(cfg)
    _model = Fusion(SimpleNamespace(**cfg)).eval()
    _model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True))


def predict_chunk(indices):
    x = torch.tensor(np.stack([_dataset[int(i)][0] for i in indices]), dtype=torch.float32)
    with torch.no_grad():
        pred = _model(x).numpy()
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite prediction')
    return indices, pred


def metrics(pred, true, horizons=(1, 2, 5, 10)):
    error = pred.astype(np.float64) - true.astype(np.float64)
    result = {'point': {}, 'prefix': {}}
    for h in horizons:
        for mode, part in [('point', error[:, h-1:h]), ('prefix', error[:, :h])]:
            result[mode][str(h)] = dict(mse=float(np.mean(part**2)),
                mae=float(np.mean(abs(part))), per_channel_mse=np.mean(part**2, axis=(0, 1)).tolist(),
                per_channel_mae=np.mean(abs(part), axis=(0, 1)).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=64, help='0 means full validation')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.windows < 0 or args.workers < 1:
        parser.error('Invalid window count or worker count')
    torch.set_num_threads(1)
    start = time.time()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.comparison / 'fusion_config.json').read_text())
    weights = args.comparison / 'fusion_epoch1.pth'
    patch_weights = args.comparison / 'baseline_checkpoints/PatchTST_first_epoch/checkpoint.pth'
    val = dataset(cfg)
    count = args.windows or len(val)
    if count > len(val):
        raise ValueError('Too many requested windows')
    indices = np.linspace(0, len(val) - 1, count, dtype=int)
    targets = np.stack([val[int(i)][1][-cfg['pred_len']:] for i in indices]).astype(np.float32)
    protocol = dict(split='validation', windows=count, total_validation_windows=len(val),
        window_indices=indices.tolist(), seq_len=cfg['seq_len'], trained_pred_len=cfg['pred_len'],
        horizons=[1, 2, 5, 10], epoch=1, normalization='training-split scaler, all 7 channels',
        point='Only the forecast at offset h', prefix='Average across offsets 1 through h',
        newly_trained=False, workers=args.workers,
        checkpoint_sha256={name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for name, p in [('CEEMDANPatchTST', weights), ('PatchTST', patch_weights)]})
    expected = json.loads((args.comparison / 'protocol.json').read_text())['fusion_checkpoint_sha256']
    assert protocol['checkpoint_sha256']['CEEMDANPatchTST'] == expected
    atomic_json(out / 'protocol.json', protocol)
    np.save(out / 'true.npy', targets)
    patch = PatchTST(SimpleNamespace(**cfg)).eval()
    patch.load_state_dict(torch.load(patch_weights, map_location='cpu', weights_only=True))
    baseline = []
    with torch.no_grad():
        for begin in range(0, count, cfg['batch_size']):
            x = torch.tensor(np.stack([val[int(i)][0] for i in indices[begin:begin + cfg['batch_size']]]), dtype=torch.float32)
            baseline.append(patch(x).numpy())
    baseline = np.concatenate(baseline)
    np.save(out / 'PatchTST_pred.npy', baseline)
    # A serial control checks the actual multiprocessing results, including
    # batch grouping, seed reset, model reload and CPU calculation.
    init_worker(cfg, weights)
    first = indices[:cfg['batch_size']].tolist()
    _, serial = predict_chunk(first)
    predictions = {}
    chunks = [indices[i:i + cfg['batch_size']].tolist() for i in range(0, count, cfg['batch_size'])]
    atomic_json(out / 'status.json', dict(status='running', completed=0, windows=count))
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
                                     initargs=(cfg, weights)) as pool:
        for ids, pred in pool.imap_unordered(predict_chunk, chunks, chunksize=1):
            predictions.update({int(i): row for i, row in zip(ids, pred)})
            atomic_json(out / 'status.json', dict(status='running', completed=len(predictions),
                        windows=count, elapsed_seconds=time.time()-start))
    fusion = np.stack([predictions[int(i)] for i in indices])
    np.testing.assert_allclose(fusion[:len(first)], serial, rtol=1e-6, atol=1e-6)
    np.save(out / 'CEEMDANPatchTST_pred.npy', fusion)
    result = dict(CEEMDANPatchTST=metrics(fusion, targets), PatchTST=metrics(baseline, targets))
    atomic_json(out / 'metrics.json', result)
    sections = []
    for mode, heading in [('prefix', '前 h 步平均误差'), ('point', '仅未来第 h 步误差')]:
        rows = []
        for h in [1, 2, 5, 10]:
            f, b = result['CEEMDANPatchTST'][mode][str(h)], result['PatchTST'][mode][str(h)]
            delta = (f['mse']/b['mse']-1)*100
            rows.append(f"| {h} | {f['mse']:.6f} | {b['mse']:.6f} | {delta:+.2f}% | {f['mae']:.6f} | {b['mae']:.6f} |")
        sections.append(f'## {heading}\n\n| h | 融合 MSE | PatchTST MSE | 融合 MSE 变化 | 融合 MAE | PatchTST MAE |\n'
                        '|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows))
    report = (f'# 短预测步长对比\n\nETTh1，7 变量、过去 96 小时输入；两个模型均为已训练一轮的 96 步输出权重，'
              f'本次没有重新训练。从完整验证区间均匀抽取 {count}/{len(val)} 个相同窗口，'
              '误差为训练段标准化后的所有变量平均值；一个步长为一小时。'
              '不是分别以 pred_len=1/2/5/10 重训的模型，也不是测试集结果。\n\n'+
              '\n\n'.join(sections)+'\n\n变化为融合 MSE / PatchTST MSE - 1，负值代表融合更好。'
              '四个步长使用同一组预测起点。完整预测、标签和窗口索引已保存，'
              '并行结果已与一个串行批次核对一致。单种子首轮权重及抽样验证只支持阶段性比较。\n')
    (out / 'REPORT.zh.md').write_text(report)
    atomic_json(out / 'status.json', dict(status='complete', windows=count,
        elapsed_seconds=time.time()-start, serial_parallel_agree=True))
    print(report, flush=True)


if __name__ == '__main__':
    main()
