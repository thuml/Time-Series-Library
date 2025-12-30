# iTransformerDiffusion 技术文档

> 本文档记录 iTransformer + CRD-Net 概率时序预测项目的详细技术讲解。

---

## 目录

- [项目概览](#项目概览)
- [模块一：数据层](#模块一数据层---数据加载与预处理)

---

## 项目概览

| 项目名称 | iTransformerDiffusion (iTransformer + CRD-Net) |
|---------|------------------------------------------------|
| 任务类型 | 概率多变量时序预测 (Probabilistic Multivariate Time Series Forecasting) |
| 核心创新 | 将 iTransformer 作为条件编码器，结合条件残差扩散网络生成概率预测 |
| 输出形式 | 预测分布的均值、标准差、多样本采样 |
| 硬件要求 | NVIDIA GPU (8GB+ VRAM)，支持 AMP 混合精度训练 |

### 模块划分总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    iTransformerDiffusion 项目架构                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │  数据层     │ → │  模型层      │ → │     实验层          │   │
│  │             │   │             │   │                     │   │
│  │ DataLoader  │   │ iTransformer│   │ exp_diffusion_      │   │
│  │ Dataset     │   │ CRD-Net     │   │ forecast.py         │   │
│  │ 数据增强    │   │ Diffusion   │   │ 两阶段训练          │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│         │                 │                    │                │
│         ▼                 ▼                    ▼                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │  工具层     │   │  采样层      │   │     评估层          │   │
│  │             │   │             │   │                     │   │
│  │ metrics.py  │   │ DDPM采样    │   │ MSE/MAE/CRPS        │   │
│  │ tools.py    │   │ DDIM加速    │   │ Calibration         │   │
│  │ masking.py  │   │ 批量并行    │   │ 可视化              │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心文件结构

```
Time-Series-Library/
├── run.py                          # 统一入口，参数解析
├── exp/
│   ├── exp_basic.py                # 基类，模型注册表
│   └── exp_diffusion_forecast.py   # 扩散预测实验类 ★核心
├── models/
│   ├── iTransformerDiffusion.py    # 主模型 ★核心
│   └── GaussianDiffusion.py        # 扩散过程工具
├── layers/
│   ├── Diffusion_layers.py         # CRD-Net网络层 ★核心
│   ├── Embed.py                    # 嵌入层
│   ├── SelfAttention_Family.py     # 注意力机制
│   └── Transformer_EncDec.py       # Transformer组件
├── data_provider/
│   ├── data_factory.py             # 数据工厂
│   └── data_loader.py              # 数据加载器
├── utils/
│   └── metrics.py                  # 评估指标 (含CRPS)
└── scripts/
    └── diffusion_forecast/         # 实验脚本
```

### 关键参数设置总览

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **模型** | d_model | 128 | Transformer隐藏维度 |
| | d_ff | 128 | 前馈网络维度 |
| | n_heads | 8 | 注意力头数 |
| | e_layers | 2 | 编码器层数 |
| **扩散** | diffusion_steps | 1000 | 扩散总步数 T |
| | beta_schedule | cosine | 噪声调度策略 |
| | cond_dim | 256 | 条件嵌入维度 |
| **训练** | stage1_epochs | 30 | 阶段1训练轮数 |
| | stage2_epochs | 20 | 阶段2训练轮数 |
| | batch_size | 8 (8GB) | 批次大小 |
| | learning_rate | 1e-4 | 学习率 |
| **采样** | n_samples | 50 | 采样数量 |
| | ddim_steps | 20 | DDIM加速步数 |
| | chunk_size | 5 | 批量采样块大小 |
| **优化** | use_amp | True | 混合精度训练 |

### 实验设计总览

| 项目 | 设置 |
|------|------|
| 数据集 | ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL |
| 划分比例 | 训练:验证:测试 = 6:2:2 (ETT) 或 7:1:2 (其他) |
| 预测长度 | 96, 192, 336, 720 |
| 评估指标 | MSE, MAE, CRPS, Calibration (50%, 90%) |
| 对比基线 | iTransformer, PatchTST, TimeMixer, Autoformer |

---

## 模块一：数据层 - 数据加载与预处理

### 1.1 数据层架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                         数据层 (Data Layer)                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   原始CSV文件                    数据工厂                           │
│   ┌─────────┐                 ┌─────────────┐                     │
│   │ ETTh1   │                 │ data_factory│                     │
│   │ .csv    │ ──────────────→ │   .py       │                     │
│   └─────────┘                 └──────┬──────┘                     │
│                                      │                            │
│                        ┌─────────────┼─────────────┐              │
│                        ▼             ▼             ▼              │
│                 ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│                 │Dataset   │  │Dataset   │  │Dataset   │          │
│                 │_ETT_hour │  │_ETT_min  │  │_Custom   │          │
│                 └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│                      │             │             │                │
│                      └─────────────┼─────────────┘                │
│                                    ▼                              │
│                        ┌───────────────────┐                      │
│                        │   DataLoader      │                      │
│                        │ (PyTorch批量加载)  │                      │
│                        └─────────┬─────────┘                      │
│                                  ▼                                │
│                        ┌───────────────────┐                      │
│                        │ (seq_x, seq_y,    │                      │
│                        │  seq_x_mark,      │                      │
│                        │  seq_y_mark)      │                      │
│                        └───────────────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心文件说明

| 文件 | 路径 | 功能 |
|------|------|------|
| data_factory.py | `data_provider/data_factory.py` | 数据工厂，统一接口 |
| data_loader.py | `data_provider/data_loader.py` | 各类数据集实现 |

### 1.3 数据工厂 (`data_factory.py`)

```python
# 数据集注册表 - 字符串名称映射到具体Dataset类
data_dict = {
    'ETTh1': Dataset_ETT_hour,    # 小时级ETT数据
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,  # 分钟级ETT数据
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,      # 自定义数据集
    ...
}

def data_provider(args, flag):
    """
    统一数据提供接口

    参数:
        args: 命令行参数对象
        flag: 'train' / 'val' / 'test'

    返回:
        data_set: Dataset对象
        data_loader: DataLoader对象
    """
    Data = data_dict[args.data]  # 根据名称获取Dataset类

    # 时间编码方式: 0=手工特征, 1=timeF自动特征
    timeenc = 0 if args.embed != 'timeF' else 1

    # 关键设置
    shuffle_flag = False if flag == 'test' else True  # 测试集不打乱

    # 创建Dataset，传入size=[seq_len, label_len, pred_len]
    data_set = Data(
        args=args,
        root_path=args.root_path,      # 数据根目录
        data_path=args.data_path,      # CSV文件名
        flag=flag,                     # train/val/test
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,        # M/S/MS
        ...
    )

    # 创建DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
```

### 1.4 ETT数据集实现 (`Dataset_ETT_hour`)

#### 1.4.1 初始化与数据划分

```python
class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, ...):
        # 序列长度配置
        self.seq_len = size[0]    # 输入序列长度 (默认96)
        self.label_len = size[1]  # 标签长度/decoder起始token (默认48)
        self.pred_len = size[2]   # 预测长度 (默认96)

        # 数据集类型映射
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
```

#### 1.4.2 数据划分边界 (ETT小时级)

```python
def __read_data__(self):
    # ETT数据集固定划分: 12个月训练, 4个月验证, 4个月测试
    # 计算: 12*30*24 = 8640小时 (训练)
    #       4*30*24 = 2880小时 (验证/测试)

    border1s = [0,
                12*30*24 - self.seq_len,           # 验证集起点
                12*30*24 + 4*30*24 - self.seq_len] # 测试集起点
    border2s = [12*30*24,                          # 训练集终点
                12*30*24 + 4*30*24,                # 验证集终点
                12*30*24 + 8*30*24]                # 测试集终点
```

**数据划分可视化：**
```
时间轴: |←————— 12个月 ——————→|←—— 4个月 ——→|←—— 4个月 ——→|
        |        训练集        |    验证集    |    测试集    |
样本数:         8640                2880           2880
比例:           60%                 20%            20%
```

#### 1.4.3 标准化处理

```python
# 关键: 只用训练集数据拟合scaler，防止数据泄露
if self.scale:
    train_data = df_data[border1s[0]:border2s[0]]  # 只取训练集
    self.scaler.fit(train_data.values)              # 拟合
    data = self.scaler.transform(df_data.values)    # 变换全部数据
```

**StandardScaler 公式:**

$$x_{scaled} = \frac{x - \mu_{train}}{\sigma_{train}}$$

### 1.5 时间特征编码

#### 1.5.1 手工特征 (timeenc=0)

```python
if self.timeenc == 0:
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
    data_stamp = df_stamp.drop(['date'], 1).values
    # 输出维度: [T, 4] - (month, day, weekday, hour)
```

#### 1.5.2 自动时间特征 (timeenc=1, timeF)

```python
elif self.timeenc == 1:
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
    data_stamp = data_stamp.transpose(1, 0)
    # 输出维度: [T, d_time] - 根据freq自动生成正弦/余弦特征
```

### 1.6 样本获取 (`__getitem__`)

```python
def __getitem__(self, index):
    """滑动窗口采样"""
    s_begin = index
    s_end = s_begin + self.seq_len        # 输入序列结束
    r_begin = s_end - self.label_len      # 标签序列开始 (有重叠)
    r_end = r_begin + self.label_len + self.pred_len  # 标签序列结束

    seq_x = self.data_x[s_begin:s_end]           # [seq_len, n_vars]
    seq_y = self.data_y[r_begin:r_end]           # [label_len+pred_len, n_vars]
    seq_x_mark = self.data_stamp[s_begin:s_end]  # [seq_len, d_time]
    seq_y_mark = self.data_stamp[r_begin:r_end]  # [label_len+pred_len, d_time]

    return seq_x, seq_y, seq_x_mark, seq_y_mark
```

**滑动窗口示意图：**
```
时间轴:    t0  t1  t2  ... t95 t96 ... t143 t144 ... t191
           |←—— seq_len=96 ——→|
                         |←— label_len=48 —→|
                                   |←—— pred_len=96 ——→|

seq_x:     [t0 ————————————— t95]           输入序列
seq_y:              [t48 ————————————— t191] 目标序列(含重叠)

重叠部分:           [t48 ——— t95]  decoder起始token
预测部分:                   [t96 ——— t191] 需要预测的部分
```

### 1.7 数据增强 (可选)

```python
# 仅在训练集且开启增强时生效
if self.set_type == 0 and self.args.augmentation_ratio > 0:
    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
        self.data_x, self.data_y, self.args
    )
```

支持的增强方法 (`utils/augmentation.py`):
- **Jittering**: 添加高斯噪声
- **Scaling**: 随机缩放
- **Permutation**: 时间段打乱
- **MagWarp**: 幅度变形
- **TimeWarp**: 时间变形

### 1.8 数据流向总结

```
┌──────────────────────────────────────────────────────────────┐
│                     完整数据流                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CSV文件                                                      │
│    │                                                         │
│    ▼                                                         │
│  DataFrame ──→ 特征选择 (M/S/MS) ──→ StandardScaler          │
│                                           │                  │
│                                           ▼                  │
│  时间戳 ──→ 时间特征编码 ──────────→ data_stamp              │
│                                           │                  │
│                                           ▼                  │
│                                    Dataset对象               │
│                                           │                  │
│                                           ▼                  │
│                              __getitem__(index)              │
│                                           │                  │
│                    ┌──────────────────────┼──────────────┐   │
│                    ▼                      ▼              ▼   │
│               seq_x                  seq_y          marks    │
│            [B,96,7]             [B,144,7]        [B,T,d]     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.9 关键参数对照表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_path` | `./dataset/ETT-small/` | 数据目录 |
| `--data_path` | `ETTh1.csv` | 数据文件名 |
| `--data` | `ETTh1` | 数据集名称 (决定Dataset类) |
| `--features` | `M` | M=多变量, S=单变量, MS=多对一 |
| `--target` | `OT` | 单变量预测目标列 |
| `--seq_len` | `96` | 输入序列长度 |
| `--label_len` | `48` | decoder起始token长度 |
| `--pred_len` | `96` | 预测长度 |
| `--embed` | `timeF` | 时间编码方式 |
| `--freq` | `h` | 时间频率 (h/t/s/m/d) |

### 1.10 ETT数据集详情

| 数据集 | 变量数 | 频率 | 总长度 | 训练/验证/测试 |
|--------|--------|------|--------|----------------|
| ETTh1 | 7 | 小时 | 17,420 | 8640/2880/2880 |
| ETTh2 | 7 | 小时 | 17,420 | 8640/2880/2880 |
| ETTm1 | 7 | 15分钟 | 69,680 | 34560/11520/11520 |
| ETTm2 | 7 | 15分钟 | 69,680 | 34560/11520/11520 |

**ETT变量含义 (电力变压器温度):**
- HUFL: 高压侧有用功负荷
- HULL: 高压侧无用功负荷
- MUFL: 中压侧有用功负荷
- MULL: 中压侧无用功负荷
- LUFL: 低压侧有用功负荷
- LULL: 低压侧无用功负荷
- OT: 油温 (通常作为预测目标)

---

*文档持续更新中...*
