"""Time-budgeted Traffic comparison with full, streaming validation and test.

Both models predict every sensor. Only validation selects checkpoints; test
labels are first accessed after both training runs finish. No large prediction
cube is allocated: retain per-window/channel errors and representative samples.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom
from models import PatchTST, TimeXer


MODULES = {'PatchTST': PatchTST, 'TimeXer': TimeXer}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def load_dataset(flag, cfg):
    dataset = Dataset_Custom(root_path=cfg.root_path, data_path='traffic.csv', flag=flag,
                             size=[cfg.seq_len, 48, cfg.pred_len], features='M',
                             target='OT', timeenc=1, freq='h')
    # Release the float64 backing array retained by the loader's split view.
    dataset.data_x = np.array(dataset.data_x, dtype=np.float32, copy=True)
    dataset.data_y = dataset.data_x
    dataset.data_stamp = np.array(dataset.data_stamp, dtype=np.float32, copy=True)
    return dataset


def forward(model, name, x, marks):
    if name == 'PatchTST':
        return model(x)
    if name == 'TimeXer':
        return model(x, marks)
    if name == 'Seasonal24':
        return x[:, -24:].repeat(1, math.ceil(x.size(1) / 24), 1)[:, :x.size(1)]
    raise ValueError(name)


def evaluate(model, name, dataset, cfg, device, output=None, deadline=None, progress=None):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    if model is not None:
        model.eval()
    square_sum = absolute_sum = raw_square_sum = raw_absolute_sum = raw_target_sum = 0.
    values = windows = 0
    channel_square = np.zeros(cfg.enc_in, dtype=np.float64)
    channel_absolute = np.zeros_like(channel_square)
    horizon_square = np.zeros(cfg.pred_len, dtype=np.float64)
    window_mse, window_mae, sample_arrays = [], [], {}
    sample_ids = {0, len(dataset) // 2, len(dataset) - 1}
    scales = dataset.scaler.scale_.astype(np.float64)
    means = dataset.scaler.mean_.astype(np.float64)
    started = time.monotonic()
    with torch.no_grad():
        for i, (x, y, marks, _) in enumerate(loader):
            if deadline is not None and time.monotonic() >= deadline:
                return None
            x, marks = x.to(device), marks.to(device)
            p = forward(model, name, x, marks).cpu().numpy().astype(np.float64)
            target = y[:, -cfg.pred_len:].numpy().astype(np.float64)
            if not np.isfinite(p).all():
                raise FloatingPointError(f'{name}: nonfinite prediction')
            error = p - target
            squared, absolute = error ** 2, np.abs(error)
            square_sum += float(squared.sum())
            absolute_sum += float(absolute.sum())
            raw_error = error * scales
            raw_square_sum += float(np.square(raw_error).sum())
            raw_absolute_sum += float(np.abs(raw_error).sum())
            raw_target_sum += float(np.abs(target * scales + means).sum())
            channel_square += squared.sum(axis=(0, 1))
            channel_absolute += absolute.sum(axis=(0, 1))
            horizon_square += squared.sum(axis=(0, 2))
            if output is not None:
                window_mse.extend(squared.mean(axis=(1, 2)).tolist())
                window_mae.extend(absolute.mean(axis=(1, 2)).tolist())
                for j in range(len(p)):
                    idx = windows + j
                    if idx in sample_ids:
                        sample_arrays[f'prediction_{idx}'] = p[j].astype(np.float32)
                        sample_arrays[f'target_{idx}'] = target[j].astype(np.float32)
                        sample_arrays[f'history_{idx}'] = x[j].cpu().numpy()
            values += error.size
            windows += len(p)
            if progress is not None and (i + 1) % 50 == 0:
                progress(windows, len(dataset))
    sync(device)
    assert windows == len(dataset)
    metrics = dict(mse=square_sum / values, mae=absolute_sum / values,
                   rmse=math.sqrt(square_sum / values), raw_mse=raw_square_sum / values,
                   raw_mae=raw_absolute_sum / values, raw_wmape=raw_absolute_sum / raw_target_sum,
                   windows=windows, values=values, seconds=time.monotonic() - started,
                   per_channel_mse=(channel_square / (windows * cfg.pred_len)).tolist(),
                   per_channel_mae=(channel_absolute / (windows * cfg.pred_len)).tolist(),
                   per_horizon_mse=(horizon_square / (windows * cfg.enc_in)).tolist())
    if output is not None:
        np.savez_compressed(output / f'{name}_errors.npz', window_mse=np.array(window_mse),
                            window_mae=np.array(window_mae), **sample_arrays)
    return metrics


def make_model(name, cfg, device):
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'mps':
        torch.mps.manual_seed(cfg.random_seed)
    return MODULES[name].Model(cfg).to(device)


def train_model(name, cfg, train, validation, out, device, seconds, absolute_deadline):
    started = time.monotonic()
    deadline = min(started + seconds, absolute_deadline)
    model = make_model(name, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1)
    history, best_mse, best_record, stale, steps, examples = [], float('inf'), None, 0, 0, 0
    train_compute_seconds = validation_seconds = 0.
    val_reserve = 65.
    stop_reason = 'time_budget'
    checkpoint_dir = out / name
    checkpoint_dir.mkdir(exist_ok=True)
    atomic_json(checkpoint_dir / 'config.json', dict(vars(cfg), model=name, device=str(device)))
    for epoch in range(1, cfg.max_epochs + 1):
        if time.monotonic() + val_reserve + 3 >= deadline:
            break
        generator = torch.Generator().manual_seed(cfg.random_seed + epoch)
        loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                            num_workers=0, generator=generator)
        model.train()
        epoch_start = time.monotonic()
        epoch_loss, epoch_examples, epoch_steps = 0., 0, 0
        partial = False
        for batch_index, (x, y, marks, _) in enumerate(loader):
            if time.monotonic() + val_reserve + 3 >= deadline:
                partial = True
                break
            x, y, marks = x.to(device), y[:, -cfg.pred_len:].to(device), marks.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward(model, name, x, marks)
            loss = (prediction - y).square().mean()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f'{name}: nonfinite training loss')
            steps += 1
            examples += len(x)
            epoch_steps += 1
            epoch_examples += len(x)
            epoch_loss += loss_value * len(x)
            if steps % 100 == 0:
                status = dict(status='training', model=name, epoch=epoch, step=steps,
                              epoch_step=batch_index + 1, epoch_batches=len(loader),
                              train_mse_so_far=epoch_loss / epoch_examples,
                              model_elapsed_seconds=time.monotonic() - started,
                              model_remaining_seconds=max(0, deadline - time.monotonic()),
                              last_validation_mse=history[-1]['validation_mse'] if history else None)
                atomic_json(out / 'status.json', status)
                print(json.dumps(status), flush=True)
        sync(device)
        train_compute_seconds += time.monotonic() - epoch_start
        if not epoch_steps:
            break
        val_start = time.monotonic()
        val = evaluate(model, name, validation, cfg, device, deadline=deadline - 1)
        spent = time.monotonic() - val_start
        validation_seconds += spent
        val_reserve = max(20., spent * 1.5 + 5)
        if val is None:
            stop_reason = 'validation_deadline'
            break
        record = dict(epoch=epoch, partial_epoch=partial, epoch_steps=epoch_steps,
                      cumulative_steps=steps, cumulative_examples=examples,
                      train_mse=epoch_loss / epoch_examples, validation_mse=val['mse'],
                      validation_mae=val['mae'], validation_windows=val['windows'],
                      lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic() - started)
        history.append(record)
        atomic_json(checkpoint_dir / 'history.json', history)
        if val['mse'] < best_mse:
            best_mse, best_record, stale = val['mse'], record, 0
            pending = checkpoint_dir / 'best.pth.tmp'
            torch.save(dict(model=model.state_dict(), config=vars(cfg), record=record), pending)
            pending.replace(checkpoint_dir / 'best.pth')
        else:
            stale += 1
        scheduler.step(val['mse'])
        print(name, 'validation', json.dumps(record), flush=True)
        if stale >= cfg.patience:
            stop_reason = 'early_stopping'
            break
        if partial:
            break
        if epoch == cfg.max_epochs:
            stop_reason = 'max_epochs'
    if best_record is None:
        raise RuntimeError(f'{name}: time budget did not permit a complete validation')
    sync(device)
    summary = dict(parameters=sum(p.numel() for p in model.parameters()),
                   model_wall_seconds=time.monotonic() - started,
                   train_compute_seconds=train_compute_seconds, validation_seconds=validation_seconds,
                   steps=steps, examples=examples, full_epochs=sum(not r['partial_epoch'] for r in history),
                   validation_checks=len(history), best=best_record, stop_reason=stop_reason,
                   checkpoint_sha256=hashlib.sha256((checkpoint_dir / 'best.pth').read_bytes()).hexdigest())
    atomic_json(checkpoint_dir / 'summary.json', summary)
    del optimizer, scheduler, model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    return summary


def write_report(out, protocol, results, elapsed):
    rows=[]
    for name in ['PatchTST', 'TimeXer', 'Seasonal24']:
        r=results[name]
        train=r.get('training', {})
        best=train.get('best', {})
        rows.append(f"| {name} | {train.get('model_wall_seconds', 0)/60:.2f} | "
                    f"{train.get('full_epochs', '—')} | {best.get('cumulative_steps', '—')} | "
                    f"{r['test']['mse']:.6f} | {r['test']['mae']:.6f} | {r['test']['rmse']:.6f} |")
    patch, timexer=results['PatchTST']['test'],results['TimeXer']['test']
    mse_improvement=(patch['mse']-timexer['mse'])/patch['mse']*100
    mae_improvement=(patch['mae']-timexer['mae'])/patch['mae']*100
    train_seconds=sum(results[n]['training']['model_wall_seconds'] for n in MODULES)
    text=f'''# Traffic：一小时内 PatchTST 与 TimeXer 对比

相同任务：全部 862 个传感器，过去 96 小时预测未来 96 小时。按时间划分训练/验证/测试 70%/10%/20%，训练段拟合标准化器。全部 {protocol['train_windows']} 个训练窗口可用；完整 {protocol['validation_windows']} 个验证窗口选模，训练全部结束后才读取完整 {protocol['test_windows']} 个测试窗口。

两个模型各获得 {protocol['per_model_budget_seconds']/60:.1f} 分钟训练及验证预算，单种子 {protocol['seed']}；使用 Apple MPS GPU、FP32，宽度 32、2 层、4 个头、FFN 64、批大小 8、Adam 初始学习率 0.0003，验证平台期降学习率。相同的每轮打乱顺序，模型的 dropout 随机流各自独立。PatchTST 保留重叠 patch（16/8、末尾填充）及 RevIN，TimeXer 保留非重叠 patch（16）及官方窗口归一化。参数量分别为 {results['PatchTST']['training']['parameters']:,} 和 {results['TimeXer']['training']['parameters']:,}。

| 模型 | 训练及验证分钟 | 完整训练轮数 | 最佳权重训练步数 | 测试 MSE ↓ | 测试 MAE ↓ | 测试 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

TimeXer 相对 PatchTST 的测试 MSE 降幅为 {mse_improvement:+.2f}%，MAE 降幅为 {mae_improvement:+.2f}%（正数表示 TimeXer 更好）。Seasonal24 为重复最近 24 小时的无训练参考。表中误差均为训练段标准化后的尺度，对所有窗口、预测步和传感器等权平均；原始尺度 MSE、MAE 和 WMAPE 也保存在 results.json。

两模型训练及验证总计 {train_seconds/60:.2f} 分钟；本脚本总耗时 {elapsed/60:.2f} 分钟（含加载、训练、完整测试和绘图；测速及准备耗时另见 timing.json）。所有终止均由时间预算、早停或预定轮数决定；最佳权重由验证 MSE 选择，未按测试结果调参。

这是相同时间预算、相同宽度下的单种子资源比较，各模型实际更新数可能不同，且参数量和原生 patch 机制不同。未做逐模型超参数搜索，也未训练到论文配置的收敛状态，不能据此断言模型普遍优劣。窗口重叠，逐窗口误差不是相互独立的重复实验。

产物：每个模型的最佳权重、配置、训练历史与校验值；完整测试逐窗口 MSE/MAE、逐通道和逐预测步误差、三个代表窗口的完整预测；对比图。为控制本机磁盘和内存用量，不保存全部窗口的三维预测张量。
'''
    (out/'REPORT.zh.md').write_text(text)
    print(text,flush=True)


def plot_results(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(14,4))
    names=['PatchTST','TimeXer','Seasonal24']
    colors=['#377eb8','#e46a39','#999999']
    for i,key in enumerate(['mse','mae']):
        axes[i].bar(names,[results[n]['test'][key] for n in names],color=colors)
        axes[i].set_title('Full-test '+key.upper()+' (lower is better)')
    for n,c in zip(names,colors):
        axes[2].plot(np.arange(1,97), results[n]['test']['per_horizon_mse'],label=n,color=c)
    axes[2].set(xlabel='Forecast horizon (hours)',ylabel='MSE',title='Error by forecast horizon')
    axes[2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,c in zip(names[:2],colors[:2]):
        hist=json.loads((out/name/'history.json').read_text())
        axes[0].plot([r['elapsed_seconds']/60 for r in hist], [r['validation_mse'] for r in hist],marker='o',label=name,color=c)
        samples=np.load(out/f'{name}_errors.npz')
        axes[1].plot(samples['prediction_0'][:,-1],label=name,color=c)
    axes[1].plot(samples['target_0'][:,-1],label='Observed',color='#333333',lw=1.5)
    axes[0].set(xlabel='Training + validation minutes',ylabel='Validation MSE',title='Validation-selected checkpoints')
    axes[1].set(xlabel='Forecast hours',ylabel='Standardized occupancy',title='First test window: sensor OT')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_and_example.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds-per-model',type=float,default=1440)
    parser.add_argument('--deadline-utc',required=True,help='Absolute UTC deadline, ISO-8601')
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    started=time.monotonic()
    wall_started=datetime.now(timezone.utc)
    end=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    absolute_deadline=started+(end-wall_started).total_seconds()
    if args.seconds_per_model <= 0 or absolute_deadline-started < 2*args.seconds_per_model+180:
        raise ValueError('Insufficient time for paired training and full test reserve')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    cfg=SimpleNamespace(root_path='./data/traffic',features='M',enc_in=862,seq_len=96,pred_len=96,
                        e_layers=2,n_heads=4,d_model=32,d_ff=64,dropout=.1,fc_dropout=.1,
                        head_dropout=.1,individual=0,patch_len=16,stride=8,padding_patch='end',
                        revin=1,affine=0,subtract_last=0,decomposition=0,kernel_size=25,
                        embed='timeF',freq='h',factor=1,activation='gelu',use_norm=1,
                        output_attention=False,batch_size=8,learning_rate=.0003,random_seed=2021,
                        patience=3,max_epochs=20)
    train,validation=load_dataset('train',cfg),load_dataset('val',cfg)
    protocol=dict(config=vars(cfg),seed=cfg.random_seed,per_model_budget_seconds=args.seconds_per_model,
                  device=str(device),torch_version=torch.__version__,train_windows=len(train),
                  validation_windows=len(validation),test_windows=3413,
                  test_access='only after both training runs finish',training_order=list(MODULES),
                  started_utc=wall_started.isoformat(),deadline_utc=end.isoformat(),
                  dataset_sha256=hashlib.sha256(Path(cfg.root_path,'traffic.csv').read_bytes()).hexdigest())
    atomic_json(args.output/'protocol.json',protocol)
    results={}
    for name in MODULES:
        summary=train_model(name,cfg,train,validation,args.output,device,args.seconds_per_model,absolute_deadline-180)
        results[name]=dict(training=summary)
        atomic_json(args.output/'results.json',results)
    del train,validation
    gc.collect()
    # No test labels or test-derived metrics have been accessed during model selection.
    test=load_dataset('test',cfg)
    assert len(test)==protocol['test_windows']
    for name in [*MODULES,'Seasonal24']:
        model=None
        if name in MODULES:
            model=make_model(name,cfg,device)
            checkpoint=torch.load(args.output/name/'best.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['model'])
        def progress(done,total):
            atomic_json(args.output/'status.json',dict(status='testing',model=name,windows=done,total=total))
        metrics=evaluate(model,name,test,cfg,device,output=args.output,deadline=absolute_deadline-10,progress=progress)
        if metrics is None:
            raise RuntimeError('Full-test deadline exceeded; incomplete metrics were not reported')
        results.setdefault(name,{})['test']=metrics
        atomic_json(args.output/'results.json',results)
        del model
        gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    plot_results(args.output,results)
    elapsed=time.monotonic()-started
    atomic_json(args.output/'timing.json',dict(script_seconds=elapsed,
                training_and_validation_seconds=sum(results[n]['training']['model_wall_seconds'] for n in MODULES),
                ended_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=end.isoformat()))
    write_report(args.output,protocol,results,elapsed)
    atomic_json(args.output/'status.json',dict(status='complete',elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    main()
