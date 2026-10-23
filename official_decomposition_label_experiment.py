"""Official PatchTST input decomposition on/off with fixed CEEMDAN labels.

The OFF controls are frozen prior runs. New ON runs use the same histories,
target windows, optimizer schedule and RAW-validation early stopping policy.
No mean matching is applied.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from label_denoise_experiment import Windows, model_config, state_hash, save_checkpoint, eval_predictions, error_metrics
from models.PatchTST import Model
from quick_validate import atomic_json
from reevaluate_denoised_targets import scores
from analyze_label_denoise import paired_interval

VARIANTS=['drop_imf1','drop_imf1_imf2']
RUN_OUTPUT=None


def configuration(cfg, channels, decomposition):
    result=model_config(cfg,channels)
    result.decomposition=bool(decomposition)
    result.kernel_size=cfg.get('kernel_size',25)
    return result


def train_open(cfg,variant,seed,inputs,labels,origins,val_loader,scale,out,baseline,deadline):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model=Model(configuration(cfg,inputs.shape[1],True)).float()
    assert model.decomposition and hasattr(model,'model_trend') and hasattr(model,'model_res')
    initialization=state_hash(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        control=Model(configuration(cfg,inputs.shape[1],False)).float()
        assert state_hash(control)==baseline['initial_state_sha256']
        for key,value in control.model.state_dict().items():
            torch.testing.assert_close(model.model_trend.state_dict()[key],value,rtol=0,atol=0)
        closed_parameters=sum(p.numel() for p in control.parameters())
    opened_parameters=sum(p.numel() for p in model.parameters())
    assert opened_parameters==2*closed_parameters
    del control
    data=Windows(inputs,labels,origins,cfg['seq_len'],cfg['pred_len'])
    loader=DataLoader(data,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(seed+10000),drop_last=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=cfg['learning_rate'],
        epochs=cfg['max_epochs'],steps_per_epoch=len(loader),pct_start=cfg['pct_start'])
    folder=out/'models'/f'{variant}_on_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    best,stale,history=math.inf,0,[]
    started=time.perf_counter()
    for epoch in range(1,cfg['max_epochs']+1):
        if time.perf_counter()>deadline:
            raise TimeoutError('Experiment time budget exceeded')
        model.train()
        label_sum,raw_sum,elements=0.,0.,0
        for x,target,raw_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction=model(x.float())
            loss=nn.functional.mse_loss(prediction,target.float())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
            label_sum+=float(loss.detach())*target.numel()
            raw_sum+=float(nn.functional.mse_loss(prediction.detach(),raw_y))*target.numel()
            elements+=target.numel()
        pred,true=eval_predictions(model,val_loader)
        val=error_metrics(pred,true,scale)
        history.append(dict(epoch=epoch,training_label_mse=label_sum/elements,
            training_raw_mse=raw_sum/elements,validation_raw_mse=val['mse'],
            elapsed_seconds=time.perf_counter()-started))
        if val['mse']<best:
            best,stale,best_epoch=val['mse'],0,epoch
            save_checkpoint(folder/'best.pt',model,optimizer,scheduler,variant=variant,seed=seed,
                epoch=epoch,config=cfg,decomposition=True,validation_raw_metrics=val,
                initial_state_sha256=initialization)
        else:stale+=1
        save_checkpoint(folder/'latest.pt',model,optimizer,scheduler,variant=variant,seed=seed,
                        epoch=epoch,config=cfg,decomposition=True)
        atomic_json(folder/'history.json',history)
        atomic_json(out/'status.json',dict(status='training',variant=variant,seed=seed,epoch=epoch,
                                         best_validation_mse=best))
        if epoch%5==0 or stale>=cfg['patience']:
            print(f'TRAIN ON {variant} seed={seed} epoch={epoch} val={val["mse"]:.6f} best={best:.6f}',flush=True)
        if stale>=cfg['patience']:break
    return dict(variant=variant,seed=seed,decomposition=True,best_epoch=best_epoch,epochs_run=len(history),
        best_validation_raw_mse=best,training_seconds=time.perf_counter()-started,
        opened_parameters=opened_parameters,closed_parameters=closed_parameters,
        trend_initialization_matches_closed=True,initial_state_sha256=initialization)


def main():
    global RUN_OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source,out=args.source,args.output
    out.mkdir(parents=True,exist_ok=True)
    RUN_OUTPUT=out
    start=time.perf_counter()
    torch.set_num_threads(1)
    old_protocol=json.loads((source/'protocol.json').read_text())
    cfg=dict(old_protocol['config'],decomposition=True,kernel_size=25)
    old_records=json.loads((source/'results.json').read_text())
    raw=pd.read_csv(cfg['data_path']).iloc[:,1:].to_numpy(dtype=np.float64)
    assert hashlib.sha256(Path(cfg['data_path']).read_bytes()).hexdigest()==old_protocol['data_sha256']
    with np.load(source/'sample_origins.npz') as a:
        train_origins,val_origins,test_origins=a['train'].copy(),a['validation'].copy(),a['test'].copy()
        mean,scale=a['train_mean'].copy(),a['train_std'].copy()
    with np.load(source/'training_labels.npz') as a:
        np.testing.assert_array_equal(a['raw'],raw[:cfg['train_end']])
        labels={v:((a[v]-mean)/scale).astype(np.float32) for v in VARIANTS}
    inputs=((raw-mean)/scale).astype(np.float32)
    test_reference_file=source/'denoised_reference_evaluation/test_reference_series.npz'
    protocol=dict(config=cfg,source_experiment=str(source.resolve()),variants=VARIANTS,
        mean_alignment=False,input_history='Identical raw histories, normalized by original training statistics',
        input_decomposition='Official moving-average trend/residual split; kernel=25; two independent PatchTST branches',
        target_decomposition='Fixed previous train-only CEEMDAN IMF removal, no coefficient multiplication',
        validation_selection='Original validation MSE for both ON and OFF, unchanged from prior runs',
        evaluation='For each removed-IMF condition, ON and OFF share exactly the same denoised test reference',
        training_windows=len(train_origins),validation_windows=len(val_origins),test_windows=len(test_origins),
        paired_sampling=True,same_seeds=True,parameter_count_changes_with_official_switch=True,
        test_reference_sha256=hashlib.sha256(test_reference_file.read_bytes()).hexdigest(),
        label_source_sha256=hashlib.sha256((source/'training_labels.npz').read_bytes()).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'protocol.json',protocol)
    np.savez(out/'sample_origins.npz',train=train_origins,validation=val_origins,test=test_origins,
             train_mean=mean,train_std=scale)
    val_loader=DataLoader(Windows(inputs,inputs,val_origins,cfg['seq_len'],cfg['pred_len']),
                          batch_size=cfg['batch_size'],num_workers=0)
    trained=[]
    for seed in cfg['seeds']:
        for variant in VARIANTS:
            baseline=next(r for r in old_records if r['variant']==variant and r['seed']==seed)
            trained.append(train_open(cfg,variant,seed,inputs,labels[variant],train_origins,
                val_loader,scale,out,baseline,start+cfg['max_seconds']))
            atomic_json(out/'training_results.json',trained)
    atomic_json(out/'selection_before_test.json',trained)
    test_loader=DataLoader(Windows(inputs,inputs,test_origins,cfg['seq_len'],cfg['pred_len']),
                           batch_size=cfg['batch_size'],num_workers=0)
    reload_difference=0.
    for row in trained:
        folder=out/'models'/f'{row["variant"]}_on_{row["seed"]}'
        saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert saved['decomposition'] is True and saved['config']['decomposition'] is True
        model=Model(configuration(cfg,raw.shape[1],True)).eval()
        model.load_state_dict(saved['model'])
        pred,true=eval_predictions(model,test_loader)
        np.testing.assert_array_equal(true,np.load(source/'test_true.npy'))
        np.save(folder/'test_pred.npy',pred)
        restored=Model(configuration(cfg,raw.shape[1],True)).eval()
        restored.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['model'])
        x=next(iter(test_loader))[0].float()
        with torch.no_grad():check=restored(x).numpy()
        np.testing.assert_allclose(check,pred[:len(x)],rtol=1e-6,atol=1e-6)
        reload_difference=max(reload_difference,float(np.max(abs(check-pred[:len(x)]))))
    with np.load(test_reference_file) as a:
        references={v:a[v].copy() for v in VARIANTS}
    offsets=test_origins-cfg['validation_end']
    target_phys={v:np.stack([a[t:t+cfg['pred_len']] for t in offsets]) for v,a in references.items()}
    target_scaled={v:((a-mean)/scale).astype(np.float32) for v,a in target_phys.items()}
    epsilon=np.maximum(1e-8,1e-6*np.mean(abs(raw[:cfg['train_end']]),axis=0))
    measurements=[]
    per_origin={v:{'off':[],'on':[]} for v in VARIANTS}
    source_scores=json.loads((source/'denoised_reference_evaluation/results.json').read_text())
    for variant in VARIANTS:
        for state in ['off','on']:
            for seed in cfg['seeds']:
                folder=(source/'models'/f'{variant}_{seed}' if state=='off' else out/'models'/f'{variant}_on_{seed}')
                pred=np.load(folder/'test_pred.npy')
                value=scores(pred,target_scaled[variant],pred.astype(np.float64)*scale+mean,target_phys[variant],epsilon)
                if state=='off':
                    old=next(r['metrics'] for r in source_scores if r['training_label']==variant and r['reference_label']==variant and r['seed']==seed)
                    for key in ['mse','mae','mape_macro_percent','wmape_macro_percent']:
                        np.testing.assert_allclose(value[key],old[key],rtol=1e-12,atol=1e-12)
                measurements.append(dict(variant=variant,decomposition=state,seed=seed,metrics=value))
                error=pred.astype(np.float64)-target_scaled[variant].astype(np.float64)
                per_origin[variant][state].append(np.mean(error**2,axis=(1,2)))
    atomic_json(out/'results.json',measurements)
    summary={}
    for variant in VARIANTS:
        summary[variant]={}
        for state in ['off','on']:
            selected=[r['metrics'] for r in measurements if r['variant']==variant and r['decomposition']==state]
            summary[variant][state]={}
            for key in ['mse','mae','mape_macro_percent','wmape_macro_percent','wmape_pooled_percent']:
                values=np.array([r[key] for r in selected])
                summary[variant][state][key]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)),seeds=values.tolist())
    uncertainty={v:paired_interval(np.stack(per_origin[v]['off']),np.stack(per_origin[v]['on'])) for v in VARIANTS}
    atomic_json(out/'paired_uncertainty.json',uncertainty)
    atomic_json(out/'summary.json',summary)
    names={'drop_imf1':'删除 IMF1','drop_imf1_imf2':'删除 IMF1＋IMF2'}
    tables=[]
    for variant in VARIANTS:
        rows=[]
        for state in ['off','on']:
            s=summary[variant][state]
            rows.append(f"| {'关闭' if state=='off' else '开启'} | {s['mse']['mean']:.6f} | {s['mae']['mean']:.6f} | {s['mape_macro_percent']['mean']:.2f}% | {s['wmape_macro_percent']['mean']:.2f}% |")
        delta=(summary[variant]['on']['mse']['mean']/summary[variant]['off']['mse']['mean']-1)*100
        ci=uncertainty[variant]['interval95_percent']
        tables.append(f"## 训练与评价标签：{names[variant]}\n\n| 官方 decomposition | MSE | MAE | MAPE | WMAPE |\n|---|---:|---:|---:|---:|\n"+'\n'.join(rows)+f'\n\n开启后 MSE 相对变化 {delta:+.2f}%；配对种子/96 小时时间区块自举近似 95% 区间 [{ci[0]:+.2f}%, {ci[1]:+.2f}%]。')
    parameters=trained[0]
    report=('# 官方输入分解开关 × CEEMDAN 去噪标签对照\n\n'
        '不进行均值对齐。ETTh1、7 变量、96→96、1,024 个训练窗口、512 个验证窗口、完整 2,785 个测试窗口、3 个相同种子。'
        '开启版本新增 6 次训练，关闭版本复用之前的固定检查点与预测；关闭指标已独立重算核对。'
        '每个标签条件内的开启/关闭模型使用同一去噪测试目标。\n\n'+'\n\n'.join(tables)+
        '\n\n## 训练、选模与指标口径\n\n'
        '- 官方 decomposition=True，kernel_size=25，输入移动平均得到趋势和残差，两个独立官方 PatchTST 分支预测后相加；没有使用自定义逐层 CEEMDAN 骨干。\n'
        '- 输入历史、CEEMDAN 标签、原始训练统计量、样本起点、每轮采样顺序、种子、Adam/OneCycle、30 轮上限和 5 轮早停均沿用关闭对照。\n'
        '- 为使本次只比较官方开关，仍按原始验证标签选择最佳 epoch，与此前关闭版本相同；本次没有按去噪验证目标重新选模。\n'
        '- 结构改变使初始化不可能全部逐权重配对；同种子下，开启模型的趋势分支初值与关闭模型一致，已核对。'
        f'关闭参数 {parameters["closed_parameters"]:,}，开启参数 {parameters["opened_parameters"]:,}。因此比较的是官方开关整体效果，无法单独归因于分解、参数增加或双分支。\n'
        '- MSE/MAE 使用同一原始训练段统计量标准化；MAPE/WMAPE 在原始单位下逐变量计算后等权平均。MAPE 的近零排除规则与此前一致，WMAPE 保留零目标处的误差。总体合并 WMAPE 另存 summary.json。\n'
        '- 测试标签来自此前已固定的测试段离线 CEEMDAN，既未重新分解，也未乘任何均值系数；这评价的是平滑目标，不是原始观测精度。\n'
        '- 自举同时重采样种子和 96 小时环形时间区块；这是单数据集、同一已观察测试集上的探索性试验。\n\n'
        f'总耗时 {(time.perf_counter()-start)/60:.2f} 分钟。检查点、预测、逐种子及逐变量指标均已保留。\n')
    (out/'REPORT.zh.md').write_text(report)
    atomic_json(out/'verification.json',dict(status='passed',official_decomposition_enabled=True,
        kernel_size=25,mean_alignment=False,unchanged_targets_verified=True,
        off_metrics_independently_reproduced=True,opened_checkpoints_reloaded=True,
        maximum_reload_absolute_difference=reload_difference,
        trend_branch_initialization_verified=True,new_models=6))
    atomic_json(out/'status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start))
    print(report,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if RUN_OUTPUT is not None:atomic_json(RUN_OUTPUT/'status.json',dict(status='failed',error=repr(e)))
        raise
