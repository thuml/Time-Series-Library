"""Two direct 10-step models, sampled targets, pure-training time budgets.

Run unattended: sequential training, validation-only selection and early stopping,
then complete held-out test evaluation and a report. No future covariates.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from compare_traffic_budget import atomic_json, load_dataset, sync
from models import PatchTST, TimeXer


NAMES = ['PatchTST', 'TimeXer']
PREFIXES = [1, 2, 5, 10]


def make_model(name, cfg, device):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'mps':
        torch.mps.manual_seed(cfg.random_seed)
    settings = dict(vars(cfg), features='MS' if name == 'TimeXer' else 'M')
    if name != 'TimeXer':
        settings.pop('target_indices')
    return {'PatchTST': PatchTST, 'TimeXer': TimeXer}[name].Model(SimpleNamespace(**settings)).to(device)


def predict(model, name, x, targets):
    # Both models see sensor histories only. TimeXer explicitly separates the
    # target set from the shared auxiliary set. PatchTST remains unmodified.
    if name == 'PatchTST':
        return model(x).index_select(2, targets)
    return model(x, None)


def evaluate(model, name, dataset, cfg, device, out=None):
    model.eval()
    targets = torch.tensor(cfg.target_indices, device=device)
    n_targets, horizon = len(cfg.target_indices), cfg.pred_len
    squares = np.zeros((horizon, n_targets), dtype=np.float64)
    absolutes = np.zeros_like(squares)
    raw_squares = np.zeros_like(squares)
    raw_absolutes = np.zeros_like(squares)
    raw_targets = np.zeros_like(squares)
    scales = dataset.scaler.scale_[cfg.target_indices]
    means = dataset.scaler.mean_[cfg.target_indices]
    windows, errors, samples = 0, [], {}
    selected = {0, len(dataset) // 2, len(dataset) - 1}
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    started = time.monotonic()
    with torch.no_grad():
        for x, y, _, _ in loader:
            output = predict(model, name, x.to(device), targets).cpu().numpy().astype(np.float64)
            truth = y[:, -horizon:, cfg.target_indices].numpy().astype(np.float64)
            if not np.isfinite(output).all():
                raise FloatingPointError(f'{name}: nonfinite evaluation output')
            error = output - truth
            squares += (error ** 2).sum(axis=0)
            absolutes += np.abs(error).sum(axis=0)
            raw_error = error * scales
            raw_squares += (raw_error ** 2).sum(axis=0)
            raw_absolutes += np.abs(raw_error).sum(axis=0)
            raw_targets += np.abs(truth * scales + means).sum(axis=0)
            if out is not None:
                errors.append(np.stack([np.mean(error[:, :k] ** 2, axis=(1, 2)) for k in PREFIXES], axis=1))
                for j in range(len(output)):
                    if windows + j in selected:
                        idx = windows + j
                        samples[f'prediction_{idx}'] = output[j].astype(np.float32)
                        samples[f'target_{idx}'] = truth[j].astype(np.float32)
                        samples[f'history_{idx}'] = x[j].index_select(1, targets.cpu()).numpy()
            windows += len(output)
    sync(device)
    assert windows == len(dataset)
    prefixes = {}
    offsets = {}
    for k in PREFIXES:
        divisor = windows * k * n_targets
        prefixes[str(k)] = dict(mse=float(squares[:k].sum() / divisor),
                               mae=float(absolutes[:k].sum() / divisor),
                               raw_mse=float(raw_squares[:k].sum() / divisor),
                               raw_mae=float(raw_absolutes[:k].sum() / divisor),
                               raw_wmape=float(raw_absolutes[:k].sum() / raw_targets[:k].sum()))
        offsets[str(k)] = dict(mse=float(squares[k-1].mean() / windows),
                              mae=float(absolutes[k-1].mean() / windows))
    metrics = dict(windows=windows, targets=n_targets, horizon=horizon,
                   seconds=time.monotonic() - started, prefixes=prefixes, offsets=offsets,
                   per_target_mse=(squares.mean(axis=0) / windows).tolist(),
                   per_offset_mse=(squares.mean(axis=1) / windows).tolist(),
                   per_offset_mae=(absolutes.mean(axis=1) / windows).tolist())
    if out is not None:
        np.savez_compressed(out / f'{name}_test_errors.npz',
                            window_prefix_mse=np.concatenate(errors),
                            target_indices=np.array(cfg.target_indices), **samples)
    return metrics


def train_one(name, cfg, train, validation, out, device, budget, interval, patience):
    folder = out / name
    folder.mkdir(exist_ok=True)
    best_path = folder / 'best.pth'
    if (folder / 'summary.json').exists():
        return json.loads((folder / 'summary.json').read_text())
    if best_path.exists() or (folder / 'history.json').exists():
        raise RuntimeError(f'Incomplete prior {name} run found; do not silently overwrite or reset its budget')
    model = make_model(name, cfg, device)
    targets = torch.tensor(cfg.target_indices, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=4)
    atomic_json(folder / 'config.json', dict(vars(cfg), model=name, features='MS' if name == 'TimeXer' else 'M',
                                           training_budget_seconds=budget, validation_interval=interval,
                                           early_stopping_patience=patience))
    history = []
    best, stale, steps, examples, epoch, epoch_step = float('inf'), 0, 0, 0, 0, 0
    train_seconds, validation_seconds, epoch_loss, epoch_examples = 0., 0., 0., 0
    best_record = None
    stop_reason = None
    started = time.monotonic()
    iterator = None
    last_validation_step = -1

    def validate():
        nonlocal best, stale, validation_seconds, best_record, last_validation_step
        metrics = evaluate(model, name, validation, cfg, device)
        validation_seconds += metrics['seconds']
        score = metrics['prefixes']['10']['mse']
        improved = score < best
        record = dict(check=len(history) + 1, epoch=epoch, epoch_step=epoch_step, step=steps,
                      examples=examples, train_seconds=train_seconds, validation_seconds=validation_seconds,
                      validation_mse=score, validation_mae=metrics['prefixes']['10']['mae'],
                      validation_windows=metrics['windows'], improved=improved,
                      train_mse_since_epoch_start=epoch_loss / max(1, epoch_examples),
                      lr=optimizer.param_groups[0]['lr'])
        if improved:
            best, stale, best_record = score, 0, record
            pending = folder / 'best.pth.tmp'
            torch.save(dict(model=model.state_dict(), config=vars(cfg), record=record), pending)
            pending.replace(best_path)
        else:
            stale += 1
        record['checks_without_improvement'] = stale
        history.append(record)
        atomic_json(folder / 'history.json', history)
        scheduler.step(score)
        last_validation_step = steps
        atomic_json(out / 'status.json', dict(status='training', model=name, **record,
                                             training_remaining_seconds=max(0, budget-train_seconds),
                                             wall_seconds=time.monotonic()-started))
        print(name, json.dumps(record), flush=True)
        # A recoverable optimizer-boundary snapshot. Replaying exact in-epoch
        # order requires the stored epoch permutation and current RNG states.
        pending = folder / 'latest.pth.tmp'
        snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), config=vars(cfg), step=steps,
                        epoch=epoch, epoch_step=epoch_step, train_seconds=train_seconds,
                        validation_seconds=validation_seconds, torch_rng=torch.get_rng_state(),
                        shuffle_seed=cfg.random_seed+epoch)
        if device.type == 'mps':
            snapshot['mps_rng'] = torch.mps.get_rng_state()
        torch.save(snapshot, pending)
        pending.replace(folder / 'latest.pth')
        model.train()

    model.train()
    while train_seconds < budget:
        if iterator is None:
            epoch += 1
            epoch_step, epoch_loss, epoch_examples = 0, 0., 0
            generator = torch.Generator().manual_seed(cfg.random_seed+epoch)
            iterator = iter(DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                                       drop_last=False, num_workers=0, generator=generator))
        sync(device)
        step_started = time.monotonic()
        try:
            x, y, _, _ = next(iterator)
        except StopIteration:
            iterator = None
            if steps != last_validation_step:
                validate()
            if stale >= patience:
                stop_reason = 'early_stopping'
                break
            continue
        x = x.to(device)
        y = y[:, -cfg.pred_len:].to(device).index_select(2, targets)
        optimizer.zero_grad(set_to_none=True)
        output = predict(model, name, x, targets)
        loss = (output-y).square().mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        sync(device)
        train_seconds += time.monotonic()-step_started
        if not math.isfinite(value):
            raise FloatingPointError(f'{name}: nonfinite training loss')
        steps += 1
        examples += len(x)
        epoch_step += 1
        epoch_examples += len(x)
        epoch_loss += value * len(x)
        if steps % interval == 0:
            validate()
            if stale >= patience:
                stop_reason = 'early_stopping'
                break
    stop_reason = stop_reason or 'training_time_budget'
    if last_validation_step != steps:
        validate()
    summary = dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                   train_seconds=train_seconds, validation_seconds=validation_seconds,
                   wall_seconds=time.monotonic()-started, epochs_started=epoch,
                   steps=steps, examples=examples, best=best_record,
                   stop_reason=stop_reason, validation_checks=len(history),
                   checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest())
    atomic_json(folder / 'summary.json', summary)
    del model, optimizer, scheduler, iterator
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return summary


def finalize(out, protocol, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for k in PREFIXES:
        a, b = (results[n]['test']['prefixes'][str(k)] for n in NAMES)
        delta = (1-b['mse']/a['mse'])*100
        rows.append(f"| 1–{k} | {a['mse']:.6f} | {b['mse']:.6f} | {a['mae']:.6f} | {b['mae']:.6f} | {delta:+.2f}% |")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in zip(NAMES, ['#377eb8', '#e46a39']):
        for ax, key in zip(axes, ['mse', 'mae']):
            ax.plot(PREFIXES, [results[name]['test']['prefixes'][str(k)][key] for k in PREFIXES],
                    marker='o', label=name, color=color)
            ax.set(xlabel='Prefix of direct 10-hour forecast', ylabel=key.upper(), xticks=PREFIXES)
            ax.legend()
    fig.tight_layout();fig.savefig(out/'comparison.png', dpi=160);plt.close(fig)
    training_rows=[]
    for name in NAMES:
        t=results[name]['training']
        training_rows.append(f"| {name} | {t['train_seconds']/60:.2f} | {t['validation_seconds']/60:.2f} | {t['steps']} | {t['stop_reason']} | {t['best']['step']} |")
    report='''# Traffic 采样目标：TimeXer MS 与 PatchTST

两个模型各训练一次，直接输出未来 10 小时，不递归滚动。下表分别评估同一最佳权重的前 1、2、5、10 步，**不是分别训练四个预测长度的实验**。

输入过去 96 小时的全部 862 个传感器。按文件顺序使用零基索引 0、10、20、…、860，共 87 个目标，其余 775 个作为 TimeXer 外生上下文。没有输入未来观测、天气、节假日或时间标记。

TimeXer 使用 MS 的多目标扩展：87 个目标分别使用自己的 patch 和全局 token，共享编码器和预测头；共享的外生 token 仅包含其余 775 个传感器，不包含选中的目标。保留默认单目标 MS 行为，本次并非原版单目标 MS。PatchTST 的模型结构保持不变，输入全部 862 个传感器，只对 87 个目标输出计算训练损失；默认 BatchNorm 在训练时存在统计耦合，推理时不融合其他通道。

按时间划分训练/验证/测试 70%/10%/20%，标准化器仅拟合训练段。两个模型的最佳权重均按完整验证集的 10 步目标 MSE 选择；两模型训练全部结束之后才计算测试指标。相同的模型宽度、优化器、初始学习率、训练窗口及每轮打乱顺序；单种子 2021。详细配置见 protocol.json。

## 完整测试结果

误差为训练段标准化尺度，对测试窗口、所选目标及前 k 个预测步等权平均。正的 MSE 降幅表示 TimeXer 更好。

| 预测步范围 | PatchTST MSE ↓ | TimeXer MSE ↓ | PatchTST MAE ↓ | TimeXer MAE ↓ | TimeXer MSE 降幅 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows) + '''

原始尺度 MSE、MAE、WMAPE、每个指定预测时刻（第 1、2、5、10 步，而非前缀平均）的误差、逐目标和逐预测步误差均保存在 results.json。完整逐窗口前缀 MSE 与三个预先指定窗口的预测保存在 *_test_errors.npz。

## 训练预算与早停

每个模型最多累计训练约 3,600 秒，计时包含取训练批次、设备传输、前向、反向、参数更新和设备同步；验证、保存、加载不计入训练预算。每 200 次更新及每轮结束执行一次完整验证，连续 15 次验证 MSE 没有严格创新低即早停。验证平台期 5 次检查后学习率减半。预算在批次边界检查，最多超过预算一个训练批次。

| 模型 | 训练分钟 | 验证分钟 | 更新次数 | 停止原因 | 最佳权重步数 |
|---|---:|---:|---:|---|---:|
''' + '\n'.join(training_rows) + '''

本次是单种子、固定配置、相同训练时间上限的实验，两个模型实际训练更新数可能不同。短于 10 步的指标来自 10 步训练目标，不能替代分别训练相应长度的结论。
'''
    report = report.replace('3,600 秒', f"{protocol['pure_training_budget_seconds_per_model']:,.0f} 秒")
    report = report.replace('每 200 次更新', f"每 {protocol['validation_interval_updates']} 次更新")
    report = report.replace('连续 15 次验证', f"连续 {protocol['early_stopping_patience']} 次验证")
    (out/'REPORT.zh.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-seconds',type=float,default=3600.)
    parser.add_argument('--validation-interval',type=int,default=200)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    if min(args.train_seconds,args.validation_interval,args.patience)<=0:
        parser.error('Budgets and intervals must be positive')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    cfg=SimpleNamespace(root_path='./data/traffic',enc_in=862,seq_len=96,pred_len=10,
                        features='M',target_indices=list(range(0,862,10)),e_layers=2,n_heads=4,
                        d_model=32,d_ff=64,dropout=.1,fc_dropout=.1,head_dropout=.1,
                        individual=0,patch_len=16,stride=8,padding_patch='end',revin=1,
                        affine=0,subtract_last=0,decomposition=0,kernel_size=25,embed='timeF',
                        freq='h',factor=1,activation='gelu',use_norm=1,output_attention=False,
                        batch_size=8,learning_rate=.0003,random_seed=2021)
    started=time.monotonic()
    try:
        atomic_json(out/'status.json',dict(status='preparing',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat()))
        train,validation=load_dataset('train',cfg),load_dataset('val',cfg)
        import pandas as pd
        columns=pd.read_csv(Path(cfg.root_path)/'traffic.csv',nrows=0).columns[1:].tolist()
        protocol=dict(config=vars(cfg),target_count=len(cfg.target_indices),
                      target_sensor_names=[columns[i] for i in cfg.target_indices],
                      exogenous_count=862-len(cfg.target_indices),train_windows=len(train),
                      validation_windows=len(validation),models_trained=2,horizon=10,evaluation_prefixes=PREFIXES,
                      pure_training_budget_seconds_per_model=args.train_seconds,
                      validation_interval_updates=args.validation_interval,early_stopping_patience=args.patience,
                      device=str(device),torch_version=torch.__version__,pid=os.getpid(),
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      dataset_sha256=hashlib.sha256(Path(cfg.root_path,'traffic.csv').read_bytes()).hexdigest(),
                      time_features_used=False,test_access='after both model selections finish')
        atomic_json(out/'protocol.json',protocol)
        results={}
        for name in NAMES:
            results[name]=dict(training=train_one(name,cfg,train,validation,out,device,
                                                 args.train_seconds,args.validation_interval,args.patience))
            atomic_json(out/'results.json',results)
        del train,validation;gc.collect()
        test=load_dataset('test',cfg)
        protocol['test_windows']=len(test)
        atomic_json(out/'protocol.json',protocol)
        for name in NAMES:
            atomic_json(out/'status.json',dict(status='testing',model=name,pid=os.getpid()))
            model=make_model(name,cfg,device)
            checkpoint=torch.load(out/name/'best.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['model'])
            results[name]['test']=evaluate(model,name,test,cfg,device,out=out)
            atomic_json(out/'results.json',results)
            del model,checkpoint;gc.collect()
            if device.type=='mps':torch.mps.empty_cache()
        finalize(out,protocol,results)
        atomic_json(out/'status.json',dict(status='complete',pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat(),
                    report=str((out/'REPORT.zh.md').resolve())))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',pid=os.getpid(),error=traceback.format_exc(),
                                          ended_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__=='__main__':
    main()
