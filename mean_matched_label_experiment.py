"""Multiply two-IMF-denoised labels by train-only, four-decimal coefficients."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from label_denoise_experiment import Windows, train_one, model_config, eval_predictions
from models.PatchTST import Model
from quick_validate import atomic_json
from reevaluate_denoised_targets import scores

NEW = 'drop_imf1_imf2_meanmatched'
VARIANTS = ['raw', 'drop_imf1', 'drop_imf1_imf2', NEW]


def mean_match_coefficients(raw, denoised):
    raw, denoised = np.asarray(raw, dtype=np.float64), np.asarray(denoised, dtype=np.float64)
    if raw.shape != denoised.shape or raw.ndim != 2:
        raise ValueError('Expected matching [training time, channels]')
    if not np.isfinite(raw).all() or not np.isfinite(denoised).all():
        raise ValueError('Nonfinite label')
    original_mean, clean_mean = raw.mean(0), denoised.mean(0)
    zero = abs(clean_mean) < 1e-12
    if np.any(zero & (abs(original_mean) >= 1e-12)):
        raise ValueError('A nonzero mean cannot be matched by multiplying a zero-mean label')
    ratio = np.divide(original_mean, clean_mean, out=np.ones_like(original_mean), where=~zero)
    coefficient = np.array([float(f'{value:.4f}') for value in ratio])
    corrected = denoised * coefficient[None, :]
    # Rounding introduces at most half a 4-decimal coefficient unit per channel.
    bound = abs(clean_mean)*.00005 + 1e-10
    assert np.all(abs(corrected.mean(0)-original_mean) <= bound)
    return coefficient, corrected


def window_means(values, origins, horizon):
    weights = np.zeros(len(values)+1, dtype=np.int64)
    np.add.at(weights, origins, 1)
    np.add.at(weights, origins+horizon, -1)
    weights = np.cumsum(weights)[:-1]
    return (values*weights[:, None]).sum(0)/weights.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, source = args.output, args.source
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(1)
    source_protocol = json.loads((source/'protocol.json').read_text())
    cfg = dict(source_protocol['config'])
    source_records = json.loads((source/'results.json').read_text())
    raw = pd.read_csv(cfg['data_path']).iloc[:, 1:].to_numpy(dtype=np.float64)
    assert hashlib.sha256(Path(cfg['data_path']).read_bytes()).hexdigest() == source_protocol['data_sha256']
    with np.load(source/'training_labels.npz') as a:
        raw_train, clean_train = a['raw'].copy(), a['drop_imf1_imf2'].copy()
    np.testing.assert_array_equal(raw_train, raw[:cfg['train_end']])
    coefficient, corrected = mean_match_coefficients(raw_train, clean_train)
    with np.load(source/'sample_origins.npz') as a:
        origins, val_origins, test_origins = a['train'].copy(), a['validation'].copy(), a['test'].copy()
        mean, scale = a['train_mean'].copy(), a['train_std'].copy()
    coefficient_rows = []
    sample_raw, sample_new = window_means(raw_train, origins, cfg['pred_len']), window_means(corrected, origins, cfg['pred_len'])
    for c, column in enumerate(source_protocol['columns']):
        m0, m1, m2 = float(raw_train[:,c].mean()), float(clean_train[:,c].mean()), float(corrected[:,c].mean())
        coefficient_rows.append(dict(variable=column, coefficient=f'{coefficient[c]:.4f}',
            raw_training_mean=m0, denoised_training_mean=m1, corrected_training_mean=m2,
            residual_relative_percent=100*(m2-m0)/abs(m0),
            actual_sampled_raw_mean=float(sample_raw[c]), actual_sampled_corrected_mean=float(sample_new[c])))
    atomic_json(out/'coefficients.json', coefficient_rows)
    cfg['label_mean_coefficients'] = coefficient.tolist()
    protocol = dict(config=cfg, source_experiment=str(source.resolve()),
        coefficient_formula='round(raw_training_mean / denoised_training_mean, 4)',
        fit_interval=[0,cfg['train_end']], coefficients_fit_on='Whole training segment before target-window sampling',
        coefficient_application='Multiply original-unit labels, THEN use original raw training scaler',
        inputs='Unchanged raw histories', training_windows=len(origins), validation_windows=len(val_origins),
        test_windows=len(test_origins), coefficients=coefficient_rows,
        validation_selection='Original raw validation labels, same as prior controls',
        test_coefficients='Reuse fixed training coefficients; never fit to test means',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'protocol.json',protocol)
    np.savez_compressed(out/'training_labels.npz', raw=raw_train, denoised=clean_train,
                        mean_matched=corrected, coefficients=coefficient)
    np.savez(out/'sample_origins.npz',train=origins,validation=val_origins,test=test_origins,
             train_mean=mean,train_std=scale)
    inputs = ((raw-mean)/scale).astype(np.float32)
    labels = ((corrected-mean)/scale).astype(np.float32)
    val_loader = DataLoader(Windows(inputs,inputs,val_origins,cfg['seq_len'],cfg['pred_len']),
                            batch_size=cfg['batch_size'],num_workers=0)
    new_records = []
    for seed in cfg['seeds']:
        record = train_one(cfg, NEW, seed, inputs, labels, origins, val_loader, scale,
                           out, start+cfg['max_seconds'])
        baseline = next(r for r in source_records if r['variant']=='raw' and r['seed']==seed)
        assert record['initial_state_sha256'] == baseline['initial_state_sha256']
        new_records.append(record)
        atomic_json(out/'training_results.json',new_records)
    atomic_json(out/'selection_before_test.json',dict(
        basis='All new checkpoints fixed using original validation MSE before any new test evaluation',
        runs=[dict(seed=r['seed'],best_epoch=r['best_epoch'],validation_mse=r['validation']['mse']) for r in new_records]))
    test_loader = DataLoader(Windows(inputs,inputs,test_origins,cfg['seq_len'],cfg['pred_len']),
                             batch_size=cfg['batch_size'],num_workers=0)
    reload_max = 0.
    for record in new_records:
        folder=out/'models'/f'{NEW}_{record["seed"]}'
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model=Model(model_config(cfg,raw.shape[1])).eval()
        model.load_state_dict(checkpoint['model'])
        pred,true=eval_predictions(model,test_loader)
        np.testing.assert_array_equal(true,np.load(source/'test_true.npy'))
        np.save(folder/'test_pred.npy',pred)
        # Independently re-create the original contiguous DataLoader first batch.
        restored=Model(model_config(cfg,raw.shape[1])).eval()
        restored.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['model'])
        x=next(iter(test_loader))[0].float()
        with torch.no_grad(): check=restored(x).numpy()
        np.testing.assert_allclose(check,pred[:len(x)],rtol=1e-6,atol=1e-6)
        reload_max=max(reload_max,float(np.max(abs(check-pred[:len(x)]))))
    ref_source=source/'denoised_reference_evaluation/test_reference_series.npz'
    with np.load(ref_source) as a:
        raw_test,clean_test=a['raw'].copy(),a['drop_imf1_imf2'].copy()
    test_series=dict(raw=raw_test,drop_imf1_imf2=clean_test,mean_matched=clean_test*coefficient)
    np.savez_compressed(out/'test_reference_series.npz',**test_series,coefficients=coefficient)
    test_mean_rows=[]
    for c,column in enumerate(source_protocol['columns']):
        a,b=float(raw_test[:,c].mean()),float(test_series['mean_matched'][:,c].mean())
        test_mean_rows.append(dict(variable=column,raw_mean=a,corrected_mean=b,relative_percent=100*(b-a)/abs(a)))
    atomic_json(out/'test_means_with_training_coefficients.json',test_mean_rows)
    epsilon=np.maximum(1e-8,1e-6*np.mean(abs(raw_train),axis=0))
    offsets=test_origins-cfg['validation_end']
    target_physical={k:np.stack([v[t:t+cfg['pred_len']] for t in offsets]) for k,v in test_series.items()}
    target_scaled={k:((v-mean)/scale).astype(np.float32) for k,v in target_physical.items()}
    np.testing.assert_array_equal(target_scaled['raw'],np.load(source/'test_true.npy'))
    all_records=source_records+new_records
    measured=[]
    for record in all_records:
        parent=out if record['variant']==NEW else source
        pred=np.load(parent/'models'/f'{record["variant"]}_{record["seed"]}'/'test_pred.npy')
        physical=pred.astype(np.float64)*scale+mean
        for reference in test_series:
            metrics=scores(pred,target_scaled[reference],physical,target_physical[reference],epsilon)
            measured.append(dict(variant=record['variant'],seed=record['seed'],reference=reference,metrics=metrics))
            if reference=='raw' and record['variant']!=NEW:
                np.testing.assert_allclose(metrics['mse'],record['test']['mse'],rtol=1e-12,atol=1e-12)
    atomic_json(out/'results.json',measured)
    summary={}
    for ref in test_series:
        summary[ref]={}
        for variant in VARIANTS:
            rows=[r['metrics'] for r in measured if r['variant']==variant and r['reference']==ref]
            summary[ref][variant]={}
            for metric in ['mse','mae','mape_macro_percent','wmape_macro_percent','wmape_pooled_percent']:
                values=np.array([r[metric] for r in rows])
                summary[ref][variant][metric]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)),seeds=values.tolist())
    atomic_json(out/'summary.json',summary)
    coefficient_table='\n'.join(f"| {r['variable']} | {r['coefficient']} | {r['raw_training_mean']:.6f} | {r['corrected_training_mean']:.6f} | {r['residual_relative_percent']:+.4f}% |" for r in coefficient_rows)
    names={'raw':'原始标签','drop_imf1':'删除 IMF1','drop_imf1_imf2':'删除 IMF1+IMF2',NEW:'删除 IMF1+IMF2，再均值校准'}
    ref_names={'raw':'原始测试序列','drop_imf1_imf2':'未校准的去噪测试标签','mean_matched':'沿用训练系数校准的去噪测试标签'}
    sections=[]
    for ref in test_series:
        rows=[]
        for variant in VARIANTS:
            s=summary[ref][variant]
            rows.append(f"| {names[variant]} | {s['mse']['mean']:.6f} | {s['mae']['mean']:.6f} | {s['mape_macro_percent']['mean']:.2f}% | {s['wmape_macro_percent']['mean']:.2f}% |")
        sections.append(f"## 评价目标：{ref_names[ref]}\n\n| 训练标签 | MSE | MAE | MAPE | WMAPE |\n|---|---:|---:|---:|---:|\n"+'\n'.join(rows))
    report=('# 两个 IMF 删除后乘系数匹配训练均值\n\n'
        '系数仅用原始训练段前 8,640 个小时拟合，四舍五入保留 4 位小数。'
        '先在原始单位下乘系数，再标准化、按相同起点采样训练。输入保持原始历史序列。\n\n'
        '| 变量 | 乘法系数 | 原始训练均值 | 校准后训练均值 | 残余变化 |\n|---|---:|---:|---:|---:|\n'+coefficient_table+
        '\n\n沿用前次 1,024 个训练窗口、512 个验证窗口、3 个种子、相同初始权重与样本顺序、相同 Adam/OneCycle/早停设置。'
        '新增三个均值校准模型；其他对照复用既有固定预测。全部使用原始验证标签选模。'
        '下表均为完整 2,785 个测试窗口、96 步、7 变量的三种子平均。\n\n'+'\n\n'.join(sections)+
        '\n\n## 口径和核对\n\n'
        '- MSE/MAE 为统一原始训练统计量标准化后的误差；MAPE/WMAPE 在原始单位下逐变量计算后等权平均。MAPE 排除极近零值，WMAPE 保留零目标处的误差；总体合并 WMAPE 另存 summary.json。\n'
        '- 测试段沿用训练系数，不按测试均值重新拟合。因此训练段均值匹配不保证测试段均值也严格匹配，详见 test_means_with_training_coefficients.json。\n'
        '- 训练段系数在采样前拟合；采样窗口重复加权后的均值可能略有不同，coefficients.json 记录了实际样本均值。\n'
        '- 新测试标签由离线分解后乘训练系数形成，属于平滑目标评价；不能把该目标上的改进解释成原始观测预测改进。\n'
        '- 这是同一数据集、已观察过测试结果后的探索性后续试验，不是新的确认性测试。\n'
        f'- 新模型训练及重评耗时 {(time.perf_counter()-start)/60:.2f} 分钟。检查点、逐模型预测、每变量指标均已保留。\n')
    (out/'REPORT.zh.md').write_text(report)
    atomic_json(out/'verification.json',dict(status='passed',coefficients_four_decimals=True,
        coefficients_train_only=True,whole_train_mean_error_within_rounding_bound=True,
        matched_sample_origins_and_initial_weights=True,raw_targets_reproduced=True,
        new_checkpoints_reloaded=True,maximum_reload_absolute_difference=reload_max))
    atomic_json(out/'status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-start))
    print(report,flush=True)


if __name__=='__main__':
    main()
