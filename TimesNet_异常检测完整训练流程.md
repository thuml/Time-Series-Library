# TimesNet异常检测任务完整训练流程详解

本文档详细记录了TimesNet模型在异常检测任务中从脚本启动到训练完成的完整函数调用流程。

## 📋 目录

- [阶段一：程序启动与初始化](#阶段一程序启动与初始化)
- [阶段二：实验类与模型初始化](#阶段二实验类与模型初始化)
- [阶段三：数据加载初始化](#阶段三数据加载初始化)
- [阶段四：训练过程](#阶段四训练过程)
- [阶段五：验证过程](#阶段五验证过程)
- [阶段六：测试与异常检测](#阶段六测试与异常检测)
- [完整调用链总结](#完整调用链总结)

---

## 🚀 阶段一：程序启动与初始化

### 1.1 脚本启动

```bash
# 典型的异常检测任务启动命令
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/PSM \
  --model_id PSM \
  --model TimesNet \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 25 \
  --c_out 25 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
```

### 1.2 程序入口 (`run.py`)

```python
# run.py:14-240
if __name__ == '__main__':
    # 1. 设置随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 2. 解析命令行参数
    parser = argparse.ArgumentParser(description='TimesNet')
    args = parser.parse_args()
    
    # 3. 设备配置
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    
    # 4. 打印参数
    print_args(args)
    
    # 5. 选择实验类 - 关键决策点
    if args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection  # ← 选择异常检测实验类
    
    # 6. 开始训练循环
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)  # ← 创建实验实例，触发初始化链
            setting = '实验名称设置...'
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)  # ← 开始训练
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)   # ← 开始测试
```

---

## 🏗️ 阶段二：实验类与模型初始化

### 2.1 实验类初始化 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:20-22
class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)  # ← 调用父类初始化
```

### 2.2 基础实验类初始化 (`exp/exp_basic.py`)

```python
# exp_basic.py:10-41
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        
        # 2.2.1 注册模型字典
        self.model_dict = {
            'TimesNet': TimesNet,  # ← 注册TimesNet模型
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            # ... 其他模型
        }
        
        # 2.2.2 获取计算设备
        self.device = self._acquire_device()
        
        # 2.2.3 构建模型并移动到设备 - 关键步骤
        self.model = self._build_model().to(self.device)  # ← 构建TimesNet模型
```

### 2.3 模型构建 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:24-28
def _build_model(self):
    # 通过模型字典获取TimesNet类并实例化
    model = self.model_dict[self.args.model].Model(self.args).float()  # ← 调用TimesNet.__init__()
    
    # 多GPU支持
    if self.args.use_multi_gpu and self.args.use_gpu:
        model = nn.DataParallel(model, device_ids=self.args.device_ids)
    return model
```

### 2.4 TimesNet模型初始化 (`models/TimesNet.py`)

```python
# TimesNet.py:76-101
def __init__(self, configs):
    super(Model, self).__init__()
    
    # 2.4.1 保存配置参数
    self.configs = configs
    self.task_name = configs.task_name  # = 'anomaly_detection'
    self.seq_len = configs.seq_len      # 输入序列长度
    self.label_len = configs.label_len  
    self.pred_len = configs.pred_len    # 异常检测为0
    
    # 2.4.2 构建TimesBlock模块列表 - 核心组件
    self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
    
    # 2.4.3 构建数据嵌入层
    self.enc_embedding = DataEmbedding(
        configs.enc_in,     # 输入特征数
        configs.d_model,    # 嵌入维度
        configs.embed,      # 嵌入类型
        configs.freq,       # 频率
        configs.dropout     # dropout率
    )
    
    # 2.4.4 构建Layer Normalization
    self.layer = configs.e_layers
    self.layer_norm = nn.LayerNorm(configs.d_model)
    
    # 2.4.5 构建任务特定的输出层
    if self.task_name == 'anomaly_detection':
        # 异常检测任务：将嵌入维度映射回原始特征维度
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
```

### 2.5 TimesBlock初始化 (`models/TimesNet.py`)

```python
# TimesNet.py:21-31
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k  # top-k个主要周期
        
        # 参数高效的设计：2D卷积序列
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )
```

---

## 📊 阶段三：数据加载初始化

### 3.1 数据加载器创建 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:30-32
def _get_data(self, flag):
    data_set, data_loader = data_provider(self.args, flag)  # ← 调用数据工厂
    return data_set, data_loader
```

### 3.2 数据工厂处理 (`data_provider/data_factory.py`)

```python
# data_factory.py:21-40
def data_provider(args, flag):
    # 根据数据集名称选择对应的数据加载器类
    Data = data_dict[args.data]  # ← 对于PSM数据集，获取PSMSegLoader
    
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,  # 数据根目录
            win_size=args.seq_len,     # 滑动窗口大小
            flag=flag,                 # train/val/test标志
        )
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
```

### 3.3 PSM数据集初始化 (`data_provider/data_loader.py`)

```python
# data_loader.py:390-406
class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        
        # 3.3.1 创建全局标准化器
        self.scaler = StandardScaler()
        
        # 3.3.2 加载训练数据并拟合标准化器
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]  # 去掉时间戳列
        data = np.nan_to_num(data)  # 处理NaN值
        
        # 关键：全局标准化 - 基于训练集统计量
        self.scaler.fit(data)  # ← 在训练集上拟合标准化器
        data = self.scaler.transform(data)  # ← 标准化训练数据
        
        # 3.3.3 加载测试数据并使用相同标准化器
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)  # ← 使用训练集统计量标准化测试集
        
        # 3.3.4 保存数据
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]  # 验证集为训练集后20%
        
        # 3.3.5 加载测试标签
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        
        print("test:", self.test.shape)
        print("train:", self.train.shape)
```

---

## 🎯 阶段四：训练过程

### 4.1 训练主循环 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:62-110
def train(self, setting):
    # 4.1.1 获取数据加载器
    train_data, train_loader = self._get_data(flag='train')
    vali_data, vali_loader = self._get_data(flag='val')
    test_data, test_loader = self._get_data(flag='test')
    
    # 4.1.2 设置训练组件
    model_optim = self._select_optimizer()  # ← Adam优化器
    criterion = self._select_criterion()    # ← MSE损失函数
    
    # 4.1.3 Early Stopping
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
    
    # 4.1.4 训练轮次循环
    for epoch in range(self.args.train_epochs):
        iter_count = 0
        train_loss = []
        
        self.model.train()  # 设置为训练模式
        epoch_time = time.time()
        
        # 4.1.5 批次训练循环
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            
            batch_x = batch_x.float().to(self.device)
            
            # 4.1.6 前向传播 - 关键步骤
            outputs = self.model(batch_x, None, None, None)  # ← 调用forward()
            
            # 4.1.7 计算重构损失
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)  # ← MSE重构误差
            train_loss.append(loss.item())
            
            # 4.1.8 反向传播和参数更新
            loss.backward()
            model_optim.step()
            
            # 4.1.9 打印训练进度
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
        
        # 4.1.10 验证和Early Stopping
        train_loss = np.average(train_loss)
        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)
        
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        
        early_stopping(vali_loss, self.model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
```

### 4.2 前向传播入口 (`models/TimesNet.py`)

```python
# TimesNet.py:202-216
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    """
    前向传播的任务分发函数
    根据task_name路由到不同的处理函数
    """
    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
    if self.task_name == 'imputation':
        dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return dec_out  # [B, L, D]
    
    if self.task_name == 'anomaly_detection':
        dec_out = self.anomaly_detection(x_enc)  # ← 异常检测路由
        return dec_out  # [B, L, D]
    
    if self.task_name == 'classification':
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]
    
    return None
```

### 4.3 异常检测核心计算 (`models/TimesNet.py`)

```python
# TimesNet.py:156-175
def anomaly_detection(self, x_enc):
    """
    异常检测的核心处理函数
    输入: x_enc [B, T, C] - 已经全局标准化的时序数据
    输出: dec_out [B, T, C] - 重构后的时序数据
    """
    
    # 4.3.1 局部标准化 (Non-stationary Transformer技术)
    means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, C] 每个序列的时间均值
    x_enc = x_enc.sub(means)  # 去均值
    stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, C] 时间标准差
    x_enc = x_enc.div(stdev)  # 标准化
    
    # 4.3.2 数据嵌入 - 将原始特征映射到高维空间
    enc_out = self.enc_embedding(x_enc, None)  # [B, T, d_model]
    
    # 4.3.3 多层TimesBlock处理 - 核心特征提取
    for i in range(self.layer):
        enc_out = self.layer_norm(self.model[i](enc_out))  # ← 逐层处理
    
    # 4.3.4 输出投影 - 映射回原始特征空间
    dec_out = self.projection(enc_out)  # [B, T, C]
    
    # 4.3.5 反标准化 - 恢复原始数据尺度
    dec_out = dec_out.mul(
        (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
    dec_out = dec_out.add(
        (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)))
    
    return dec_out
```

### 4.4 TimesBlock核心计算 (`models/TimesNet.py`)

```python
# TimesNet.py:33-67
def forward(self, x):  # TimesBlock.forward()
    """
    TimesBlock的核心处理逻辑
    1. FFT周期发现
    2. 多周期2D卷积处理
    3. 自适应聚合
    """
    B, T, N = x.size()
    
    # 4.4.1 FFT周期发现
    period_list, period_weight = FFT_for_Period(x, self.k)
    
    # 4.4.2 多周期处理
    res = []
    for i in range(self.k):
        period = period_list[i]
        
        # 4.4.3 填充到周期整数倍长度
        if (self.seq_len + self.pred_len) % period != 0:
            length = (((self.seq_len + self.pred_len) // period) + 1) * period
            padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = (self.seq_len + self.pred_len)
            out = x
        
        # 4.4.4 重塑为2D矩阵 - 关键创新
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        # 形状变化: [B, T, N] → [B, N, 周期数, 周期长度]
        
        # 4.4.5 2D卷积处理 - 捕获周期内和周期间的模式
        out = self.conv(out)  # ← Inception卷积块处理
        
        # 4.4.6 重塑回1D时序
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        res.append(out[:, :(self.seq_len + self.pred_len), :])
    
    # 4.4.7 自适应聚合 - 加权融合多个周期的结果
    res = torch.stack(res, dim=-1)  # [B, T, N, k]
    period_weight = F.softmax(period_weight, dim=1)
    period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
    res = torch.sum(res * period_weight, -1)  # 加权求和
    
    # 4.4.8 残差连接
    res = res + x
    return res
```

### 4.5 FFT周期发现 (`models/TimesNet.py`)

```python
# TimesNet.py:8-17
def FFT_for_Period(x, k=2):
    """
    使用FFT发现时序数据的主要周期
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)  # 实数FFT变换
    
    # 通过幅度谱找到主要周期
    frequency_list = abs(xf).mean(0).mean(-1)  # 平均幅度谱
    frequency_list[0] = 0  # 忽略直流分量
    _, top_list = torch.topk(frequency_list, k)  # 找到top-k频率
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 周期 = 序列长度 / 频率索引
    
    return period, abs(xf).mean(-1)[:, top_list]  # 返回周期和对应权重
```

---

## 📈 阶段五：验证过程

### 5.1 验证函数 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:41-61
def vali(self, vali_data, vali_loader, criterion):
    """
    验证函数：计算验证集上的重构损失
    用于Early Stopping和模型选择
    """
    total_loss = []
    self.model.eval()  # ← 切换到评估模式
    
    with torch.no_grad():  # 关闭梯度计算
        for i, (batch_x, _) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            
            # 5.1.1 前向传播（无梯度计算）
            outputs = self.model(batch_x, None, None, None)
            
            # 5.1.2 计算验证损失
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs.detach().cpu()
            true = batch_x.detach().cpu()
            
            loss = criterion(pred, true)
            total_loss.append(loss)
    
    total_loss = np.average(total_loss)
    self.model.train()  # ← 切换回训练模式
    return total_loss
```

---

## 🧪 阶段六：测试与异常检测

### 6.1 测试主函数 (`exp/exp_anomaly_detection.py`)

```python
# exp_anomaly_detection.py:126-208
def test(self, setting, test=0):
    """
    测试函数：执行异常检测并评估性能
    """
    test_data, test_loader = self._get_data(flag='test')
    train_data, train_loader = self._get_data(flag='train')
    
    # 6.1.1 加载最佳模型
    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    
    self.model.eval()
    self.anomaly_criterion = nn.MSELoss(reduce=False)  # 逐元素MSE
    
    # 6.1.2 训练集统计 - 建立正常数据的重构误差基线
    attens_energy = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(self.device)
            
            # 重构训练数据
            outputs = self.model(batch_x, None, None, None)
            
            # 计算重构误差
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    
    # 6.1.3 测试集异常检测
    attens_energy = []
    test_labels = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.float().to(self.device)
        
        # 重构测试数据
        outputs = self.model(batch_x, None, None, None)
        
        # 计算重构误差
        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(batch_y)
    
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    
    # 6.1.4 阈值确定 - 基于训练集和测试集的联合分布
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
    print("Threshold :", threshold)
    
    # 6.1.5 异常判定 - 二值化
    pred = (test_energy > threshold).astype(int)  # 重构误差大于阈值则为异常
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)
    
    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)
    
    # 6.1.6 检测调整 - 后处理
    gt, pred = adjustment(gt, pred)  # 调整预测结果
    
    # 6.1.7 性能评估
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision, recall, f_score))
    
    # 6.1.8 保存结果
    f = open("result_anomaly_detection.txt", 'a')
    f.write(setting + "  \n")
    f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision, recall, f_score))
    f.write('\n')
    f.write('\n')
    f.close()
```

---

## 🔄 完整调用链总结

### 调用流程图

```
📋 启动命令
    ↓
🚀 run.py:main()
    ├── 解析参数 & 设备配置
    ├── 选择Exp_Anomaly_Detection类
    └── 创建实验实例
        ↓
🏗️ Exp_Anomaly_Detection.__init__()
    └── 调用父类 Exp_Basic.__init__()
        ├── 注册模型字典
        ├── 获取计算设备
        └── 构建模型: _build_model()
            ↓
🔧 TimesNet.Model.__init__()
    ├── 保存配置参数
    ├── 创建TimesBlock模块列表
    ├── 创建数据嵌入层
    ├── 创建LayerNorm
    └── 创建输出投影层
        ↓
📊 数据初始化
    └── _get_data() → data_provider() → PSMSegLoader.__init__()
        ├── 创建StandardScaler
        ├── 加载训练数据并拟合标准化器
        ├── 全局标准化变换
        └── 加载测试数据并标准化
            ↓
🎯 训练阶段: exp.train(setting)
    ├── 获取数据加载器
    ├── 设置优化器和损失函数
    └── 训练循环:
        for epoch in range(train_epochs):
            for batch in train_loader:
                ├── 前向传播: model(batch_x)
                │   └── forward() → anomaly_detection()
                │       ├── 局部标准化
                │       ├── 数据嵌入
                │       ├── 多层TimesBlock处理
                │       │   ├── FFT周期发现
                │       │   ├── 多周期2D卷积
                │       │   ├── 自适应聚合
                │       │   └── 残差连接
                │       ├── 输出投影
                │       └── 反标准化
                ├── 计算MSE重构损失
                ├── 反向传播
                └── 参数更新
            ├── 验证: vali()
            └── Early Stopping检查
                ↓
🧪 测试阶段: exp.test(setting)
    ├── 加载最佳模型
    ├── 训练集重构误差统计
    ├── 测试集重构误差计算
    ├── 阈值确定 (百分位数)
    ├── 异常判定 (二值化)
    ├── 后处理调整
    └── 性能评估 (Precision, Recall, F1)
```

### 关键函数调用频次

| 函数 | 调用次数 | 说明 |
|------|----------|------|
| `TimesNet.__init__()` | 1次 | 程序启动时调用一次 |
| `PSMSegLoader.__init__()` | 3次 | train/val/test各一次 |
| `forward()` | N×M次 | N个epoch × M个batch |
| `anomaly_detection()` | N×M次 | 每次前向传播调用 |
| `TimesBlock.forward()` | N×M×L次 | L为层数(e_layers) |
| `FFT_for_Period()` | N×M×L次 | 每个TimesBlock都调用 |
| `PSMSegLoader.__getitem__()` | 数据总量次 | 每个样本调用一次 |

### 数据流形状变化

```
原始CSV数据 → 加载与预处理
    ↓
全局标准化: StandardScaler.transform()
    ↓
滑动窗口切分: [B, T, C] (batch_size, seq_len, features)
    ↓
局部标准化: 每个样本在时间维度标准化 [B, T, C]
    ↓
数据嵌入: [B, T, C] → [B, T, d_model]
    ↓
TimesBlock处理:
    FFT周期发现: [B, T, d_model] → 周期列表
    1D→2D变换: [B, T, d_model] → [B, d_model, 周期数, 周期长度]
    2D卷积: Inception块处理
    2D→1D变换: [B, d_model, 周期数, 周期长度] → [B, T, d_model]
    自适应聚合: 多周期结果加权融合
    ↓
输出投影: [B, T, d_model] → [B, T, C]
    ↓
反标准化: 恢复到原始数据尺度 [B, T, C]
    ↓
重构误差计算: MSE(原始, 重构) → 异常分数
```

### 关键设计原理

1. **双层标准化策略**:
   - 全局标准化: 解决特征尺度问题
   - 局部标准化: 解决非平稳时序问题

2. **TimesNet核心创新**:
   - FFT自动发现周期性
   - 1D→2D变换利用卷积捕获模式
   - 多周期自适应聚合

3. **异常检测原理**:
   - 基于重构误差的无监督方法
   - 阈值基于训练集统计确定
   - 后处理调整提高实用性

4. **端到端训练**:
   - 统一的重构损失函数
   - Early Stopping防止过拟合
   - 多GPU支持大规模训练

---

## 📝 总结

TimesNet在异常检测任务中展现了以下特点:

1. **模块化设计**: 清晰的分层架构，便于理解和扩展
2. **高效训练**: 端到端的训练流程，自动化程度高
3. **技术创新**: FFT周期发现 + 2D卷积的独特组合
4. **工程实用**: 完善的数据处理、模型保存、性能评估流程

这个完整的流程涵盖了从命令行启动到最终结果输出的每一个关键步骤，为理解和使用TimesNet提供了详细的参考。 