"""
简化的MoM测试脚本 - 直接测试不同MoM配置
"""
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入run.py的参数解析器
from run import main
import argparse

# 测试不同的MoM配置
configs = [
    ('use_mom_false', False, 10, '简单均值 (Baseline)'),
    ('use_mom_k5', True, 5, 'MoM k=5'),
    ('use_mom_k10', True, 10, 'MoM k=10 (当前)'),
    ('use_mom_k20', True, 20, 'MoM k=20'),
]

print("="*80)
print("MoM 优化测试")
print("="*80)
print()
print("注意: 这个脚本会临时修改模型代码中的use_mom和mom_k参数")
print("然后运行测试，最后恢复原始代码")
print()

# 读取原始模型代码
model_file = './models/iTransformerDiffusionDirect.py'
with open(model_file, 'r') as f:
    original_code = f.read()

results = []

for desc, use_mom, mom_k, name in configs:
    print(f"\n{'='*80}")
    print(f"测试配置: {name}")
    print(f"{'='*80}\n")

    # 修改模型代码中的默认参数
    modified_code = original_code.replace(
        'use_mom=True,',
        f'use_mom={use_mom},'
    ).replace(
        'mom_k=10,',
        f'mom_k={mom_k},'
    )

    # 写入修改后的代码
    with open(model_file, 'w') as f:
        f.write(modified_code)

    # 运行测试
    sys.argv = [
        'run.py',
        '--task_name', 'diffusion_forecast',
        '--is_training', '0',
        '--model', 'iTransformerDiffusionDirect',
        '--data', 'ETTh1',
        '--root_path', './dataset/ETT-small/',
        '--data_path', 'ETTh1.csv',
        '--features', 'M',
        '--seq_len', '96',
        '--label_len', '48',
        '--pred_len', '96',
        '--enc_in', '7',
        '--dec_in', '7',
        '--c_out', '7',
        '--d_model', '128',
        '--d_ff', '128',
        '--e_layers', '2',
        '--n_heads', '8',
        '--diffusion_steps', '1000',
        '--beta_schedule', 'cosine',
        '--parameterization', 'x0',
        '--n_samples', '100',
        '--use_ddim',
        '--ddim_steps', '50',
        '--chunk_size', '10',
        '--use_amp',
        '--des', f'MoM_{desc}'
    ]

    try:
        main()
    except SystemExit:
        pass

    # 恢复原始代码
    with open(model_file, 'w') as f:
        f.write(original_code)

print("\n" + "="*80)
print("所有测试完成!")
print("="*80)
print("\n请查看 result_diffusion_forecast.txt 文件查看结果对比")
