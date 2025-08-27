"""
诊断TimesNet重构异常检测失败的深层原因
分析为什么Contact数据集上的无监督重构方法表现不佳
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import logger
import os

class ReconstructionAnomalyDiagnostic:
    def __init__(self):
        self.results = {}
        
    def load_contact_data(self):
        """加载Contact数据"""
        logger.info("📂 加载Contact数据...")
        
        # 加载训练数据（正常样本）
        train_data = pd.read_parquet('./dataset/anomaly_filtered_Traindata/train_normal.parquet')
        
        # 加载测试数据和标签
        test_data = pd.read_parquet('./dataset/split_data/test.parquet')
        test_labels = pd.read_parquet('./dataset/anomaly_labeld_testData/test_labeled.parquet')
        
        # 模拟ContactLoader的预处理
        train_features = train_data.values[:, 1:]  # 跳过TimeStamp
        test_features = test_data.values[:, 1:]
        test_labels_array = test_labels.values[:, 1:].flatten()
        
        # 标准化
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        
        logger.info(f"训练数据形状: {train_features.shape}")
        logger.info(f"测试数据形状: {test_features.shape}")
        logger.info(f"异常比例: {test_labels_array.sum()/len(test_labels_array)*100:.2f}%")
        
        return train_features, test_features, test_labels_array, scaler
    
    def analyze_data_characteristics(self, train_data, test_data, test_labels):
        """分析数据特性对重构的影响"""
        logger.info("🔍 分析数据特性...")
        
        normal_mask = test_labels == 0
        anomaly_mask = test_labels == 1
        
        normal_data = test_data[normal_mask]
        anomaly_data = test_data[anomaly_mask]
        
        # 1. 数据复杂性分析
        logger.info("=== 数据复杂性分析 ===")
        
        # 特征间相关性
        correlation_matrix = np.corrcoef(train_data.T)
        high_corr_pairs = np.sum(np.abs(correlation_matrix) > 0.9) - train_data.shape[1]  # 排除对角线
        logger.info(f"高相关性特征对数量 (>0.9): {high_corr_pairs // 2}")
        
        # 特征方差分析
        feature_variances = np.var(train_data, axis=0)
        low_variance_features = np.sum(feature_variances < 0.01)
        logger.info(f"低方差特征数量 (<0.01): {low_variance_features}")
        
        # 数据分布偏度
        from scipy.stats import skew, kurtosis
        feature_skewness = [skew(train_data[:, i]) for i in range(train_data.shape[1])]
        high_skew_features = np.sum(np.abs(feature_skewness) > 2)
        logger.info(f"高偏度特征数量 (|skew|>2): {high_skew_features}")
        
        # 2. 正常vs异常样本的可重构性分析
        logger.info("\n=== 可重构性分析 ===")
        
        # 计算样本内部方差（复杂度）
        normal_complexity = np.mean([np.var(sample) for sample in normal_data])
        anomaly_complexity = np.mean([np.var(sample) for sample in anomaly_data])
        
        logger.info(f"正常样本平均复杂度: {normal_complexity:.6f}")
        logger.info(f"异常样本平均复杂度: {anomaly_complexity:.6f}")
        logger.info(f"复杂度比例: {anomaly_complexity/normal_complexity:.3f}")
        
        # 计算与训练集的相似性
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 随机采样计算相似性（避免内存问题）
        sample_size = min(1000, len(train_data))
        train_sample = train_data[np.random.choice(len(train_data), sample_size, replace=False)]
        
        normal_similarities = []
        anomaly_similarities = []
        
        for i in range(min(100, len(normal_data))):
            sim = cosine_similarity([normal_data[i]], train_sample)[0]
            normal_similarities.append(np.mean(sim))
            
        for i in range(min(100, len(anomaly_data))):
            sim = cosine_similarity([anomaly_data[i]], train_sample)[0]
            anomaly_similarities.append(np.mean(sim))
        
        logger.info(f"正常样本与训练集平均相似性: {np.mean(normal_similarities):.4f}")
        logger.info(f"异常样本与训练集平均相似性: {np.mean(anomaly_similarities):.4f}")
        
        self.results['data_characteristics'] = {
            'high_correlation_pairs': high_corr_pairs // 2,
            'low_variance_features': low_variance_features,
            'high_skew_features': high_skew_features,
            'normal_complexity': normal_complexity,
            'anomaly_complexity': anomaly_complexity,
            'complexity_ratio': anomaly_complexity/normal_complexity,
            'normal_similarity': np.mean(normal_similarities),
            'anomaly_similarity': np.mean(anomaly_similarities)
        }
        
    def test_simple_reconstruction_baselines(self, train_data, test_data, test_labels):
        """测试简单重构基线方法"""
        logger.info("🧪 测试简单重构基线...")
        
        results = {}
        
        # 1. 线性自编码器
        logger.info("测试线性自编码器...")
        
        class LinearAutoEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, hidden_dim)
                self.decoder = nn.Linear(hidden_dim, input_dim)
                
            def forward(self, x):
                encoded = torch.relu(self.encoder(x))
                decoded = self.decoder(encoded)
                return decoded
        
        # 训练线性自编码器
        input_dim = train_data.shape[1]
        hidden_dim = input_dim // 2
        
        model = LinearAutoEncoder(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_tensor = torch.FloatTensor(train_data)
        test_tensor = torch.FloatTensor(test_data)
        
        # 简单训练
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            reconstructed = model(train_tensor)
            loss = criterion(reconstructed, train_tensor)
            loss.backward()
            optimizer.step()
        
        # 测试重构误差
        model.eval()
        with torch.no_grad():
            test_reconstructed = model(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - test_reconstructed) ** 2, dim=1).numpy()
        
        # 计算异常检测性能
        thresholds = np.percentile(reconstruction_errors, [90, 95, 99, 99.5])
        
        for i, thresh in enumerate(thresholds):
            predictions = (reconstruction_errors > thresh).astype(int)
            # 确保标签格式正确
            test_labels_clean = test_labels.astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels_clean, predictions, average='binary')
            results[f'linear_ae_p{[90, 95, 99, 99.5][i]}'] = {
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            logger.info(f"  P{[90, 95, 99, 99.5][i]}阈值: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        
        # 2. PCA重构
        logger.info("测试PCA重构...")
        from sklearn.decomposition import PCA
        
        for n_components in [10, 20, input_dim//2]:
            if n_components >= input_dim:
                continue
                
            pca = PCA(n_components=n_components)
            pca.fit(train_data)
            
            # 重构测试数据
            test_transformed = pca.transform(test_data)
            test_reconstructed_pca = pca.inverse_transform(test_transformed)
            
            pca_errors = np.mean((test_data - test_reconstructed_pca) ** 2, axis=1)
            
            thresh = np.percentile(pca_errors, 95)
            predictions = (pca_errors > thresh).astype(int)
            test_labels_clean = test_labels.astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(test_labels_clean, predictions, average='binary')
            
            results[f'pca_{n_components}'] = {
                'explained_variance': pca.explained_variance_ratio_.sum(),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            logger.info(f"  PCA-{n_components}: F1={f1:.4f}, 解释方差={pca.explained_variance_ratio_.sum():.3f}")
        
        # 3. 统计基线：马哈拉诺比斯距离
        logger.info("测试马哈拉诺比斯距离...")
        
        # 计算训练数据的均值和协方差
        train_mean = np.mean(train_data, axis=0)
        train_cov = np.cov(train_data.T)
        
        # 加入正则化避免奇异矩阵
        train_cov_reg = train_cov + np.eye(train_cov.shape[0]) * 1e-6
        train_cov_inv = np.linalg.inv(train_cov_reg)
        
        # 计算马哈拉诺比斯距离
        def mahalanobis_distance(x, mean, cov_inv):
            diff = x - mean
            return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        
        mahal_distances = mahalanobis_distance(test_data, train_mean, train_cov_inv)
        
        thresh = np.percentile(mahal_distances, 95)
        predictions = (mahal_distances > thresh).astype(int)
        test_labels_clean = test_labels.astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels_clean, predictions, average='binary')
        
        results['mahalanobis'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        logger.info(f"  马哈拉诺比斯距离: F1={f1:.4f}")
        
        self.results['baseline_methods'] = results
        
        # 分析为什么重构方法失败
        logger.info("\n=== 重构失败原因分析 ===")
        
        # 检查重构误差分布
        test_labels_clean = test_labels.astype(int)
        normal_errors = reconstruction_errors[test_labels_clean == 0]
        anomaly_errors = reconstruction_errors[test_labels_clean == 1]
        
        logger.info(f"正常样本重构误差: 均值={np.mean(normal_errors):.6f}, 标准差={np.std(normal_errors):.6f}")
        logger.info(f"异常样本重构误差: 均值={np.mean(anomaly_errors):.6f}, 标准差={np.std(anomaly_errors):.6f}")
        logger.info(f"误差分离度: {(np.mean(anomaly_errors) - np.mean(normal_errors)) / np.std(normal_errors):.3f}")
        
        # 如果异常样本的重构误差不显著高于正常样本，说明模型学习了错误的表示
        if np.mean(anomaly_errors) <= np.mean(normal_errors) * 1.1:
            logger.warning("⚠️ 异常样本的重构误差与正常样本相近，模型可能过度泛化")
            
    def analyze_timesnet_specific_issues(self, train_data, test_data, test_labels):
        """分析TimesNet特定的问题"""
        logger.info("🔍 分析TimesNet特定问题...")
        
        # 1. 时序长度问题
        logger.info("=== 时序长度分析 ===")
        
        # Contact数据的seq_len通常是100，检查这个长度是否合适
        seq_lens_to_test = [50, 100, 200]
        
        for seq_len in seq_lens_to_test:
            if seq_len > len(test_data):
                continue
                
            # 模拟时序窗口
            num_windows = len(test_data) - seq_len + 1
            
            # 计算时序窗口内的方差（复杂度）
            window_variances = []
            for i in range(min(100, num_windows)):  # 只计算前100个窗口
                window = test_data[i:i+seq_len]
                window_variance = np.var(window)
                window_variances.append(window_variance)
            
            logger.info(f"  seq_len={seq_len}: 窗口平均方差={np.mean(window_variances):.6f}")
        
        # 2. 周期性分析
        logger.info("\n=== 周期性分析 ===")
        
        # 检查数据中是否存在明显的周期性模式
        from scipy.fft import fft, fftfreq
        
        # 对前几个特征进行FFT分析
        for feat_idx in range(min(3, train_data.shape[1])):
            signal = train_data[:min(1440, len(train_data)), feat_idx]  # 最多取1天的数据
            
            # FFT分析
            fft_values = fft(signal)
            freqs = fftfreq(len(signal))
            
            # 找到主要频率成分
            magnitude = np.abs(fft_values)
            dominant_freq_idx = np.argsort(magnitude[1:len(magnitude)//2])[-3:]  # 前3个主要频率
            
            logger.info(f"  特征{feat_idx}的主要周期: {[1/freqs[idx+1] if freqs[idx+1] != 0 else 'inf' for idx in dominant_freq_idx]}")
        
        # 3. 特征重要性分析
        logger.info("\n=== 特征重要性分析 ===")
        
        # 使用方差和区分力分析特征重要性
        feature_importance = {}
        
        normal_data = test_data[test_labels == 0]
        anomaly_data = test_data[test_labels == 1]
        
        for i in range(train_data.shape[1]):
            # 方差（信息量）
            variance = np.var(train_data[:, i])
            
            # 区分力（正常vs异常）
            normal_mean = np.mean(normal_data[:, i])
            anomaly_mean = np.mean(anomaly_data[:, i])
            pooled_std = np.sqrt((np.var(normal_data[:, i]) + np.var(anomaly_data[:, i])) / 2)
            discrimination = abs(normal_mean - anomaly_mean) / (pooled_std + 1e-8)
            
            feature_importance[i] = {
                'variance': variance,
                'discrimination': discrimination,
                'combined_score': variance * discrimination
            }
        
        # 排序特征
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        logger.info("Top 5 重要特征:")
        for feat_idx, metrics in sorted_features[:5]:
            logger.info(f"  特征{feat_idx}: 方差={metrics['variance']:.4f}, 区分力={metrics['discrimination']:.4f}")
        
        # 4. 数据质量问题
        logger.info("\n=== 数据质量问题 ===")
        
        quality_issues = []
        
        # 检查异常值
        outlier_ratios = []
        for i in range(train_data.shape[1]):
            Q1 = np.percentile(train_data[:, i], 25)
            Q3 = np.percentile(train_data[:, i], 75)
            IQR = Q3 - Q1
            outliers = np.sum((train_data[:, i] < Q1 - 1.5*IQR) | (train_data[:, i] > Q3 + 1.5*IQR))
            outlier_ratio = outliers / len(train_data)
            outlier_ratios.append(outlier_ratio)
        
        avg_outlier_ratio = np.mean(outlier_ratios)
        logger.info(f"平均异常值比例: {avg_outlier_ratio*100:.2f}%")
        
        if avg_outlier_ratio > 0.1:
            quality_issues.append("训练数据包含过多异常值")
        
        # 检查数据平稳性
        non_stationary_features = 0
        for i in range(min(10, train_data.shape[1])):
            # 简单的平稳性检查：前后半段均值差异
            first_half = train_data[:len(train_data)//2, i]
            second_half = train_data[len(train_data)//2:, i]
            
            mean_diff = abs(np.mean(first_half) - np.mean(second_half))
            std_pooled = np.sqrt((np.var(first_half) + np.var(second_half)) / 2)
            
            if mean_diff > 2 * std_pooled:
                non_stationary_features += 1
        
        logger.info(f"可能非平稳的特征数: {non_stationary_features}/10")
        
        if non_stationary_features > 5:
            quality_issues.append("数据可能非平稳")
        
        self.results['timesnet_analysis'] = {
            'feature_importance': dict(sorted_features[:10]),
            'avg_outlier_ratio': avg_outlier_ratio,
            'non_stationary_features': non_stationary_features,
            'quality_issues': quality_issues
        }
        
    def propose_solutions(self):
        """提出解决方案"""
        logger.info("💡 提出解决方案...")
        
        solutions = []
        
        # 基于分析结果提出解决方案
        data_char = self.results['data_characteristics']
        timesnet_analysis = self.results['timesnet_analysis']
        
        # 1. 数据预处理改进
        if data_char['high_correlation_pairs'] > 50:
            solutions.append("使用PCA或特征选择减少高相关性特征")
            
        if data_char['low_variance_features'] > 5:
            solutions.append("移除低方差特征")
            
        if data_char['complexity_ratio'] < 1.2:
            solutions.append("异常样本复杂度不足，考虑重新定义异常标准")
            
        # 2. 模型改进
        if data_char['anomaly_similarity'] > 0.8:
            solutions.append("异常样本与训练集相似度过高，考虑使用更复杂的模型")
            
        # 3. 训练策略改进
        best_baseline_f1 = max([result['f1'] for result in self.results['baseline_methods'].values()])
        if best_baseline_f1 > 0.2:
            solutions.append(f"简单基线方法表现更好(F1={best_baseline_f1:.3f})，TimesNet可能过于复杂")
            
        # 4. 数据质量改进
        if timesnet_analysis['quality_issues']:
            solutions.extend([f"解决数据质量问题: {issue}" for issue in timesnet_analysis['quality_issues']])
            
        # 5. 阈值策略改进
        solutions.append("使用动态阈值或多阈值融合策略")
        solutions.append("考虑使用训练数据的重构误差分布来设定阈值")
        
        logger.info("建议的解决方案:")
        for i, solution in enumerate(solutions, 1):
            logger.info(f"  {i}. {solution}")
            
        self.results['solutions'] = solutions
        
    def generate_comprehensive_report(self):
        """生成综合报告"""
        logger.info("📋 生成综合分析报告...")
        
        report = f"""
# TimesNet重构异常检测失败原因分析报告

## 🎯 核心发现

### 数据特性问题
- 高相关性特征对: {self.results['data_characteristics']['high_correlation_pairs']}
- 低方差特征数: {self.results['data_characteristics']['low_variance_features']}
- 异常样本复杂度比例: {self.results['data_characteristics']['complexity_ratio']:.3f}
- 异常样本相似性: {self.results['data_characteristics']['anomaly_similarity']:.3f}

### 基线方法性能
"""
        
        for method, result in self.results['baseline_methods'].items():
            if 'f1' in result:
                report += f"- {method}: F1={result['f1']:.4f}\n"
        
        report += f"""
### TimesNet特定问题
- 平均异常值比例: {self.results['timesnet_analysis']['avg_outlier_ratio']*100:.2f}%
- 非平稳特征数: {self.results['timesnet_analysis']['non_stationary_features']}/10
- 数据质量问题: {', '.join(self.results['timesnet_analysis']['quality_issues']) if self.results['timesnet_analysis']['quality_issues'] else '无'}

## 🔍 失败原因分析

### 1. 数据本身的问题
- **高维度低区分度**: 27个特征中可能存在大量冗余信息
- **异常样本不够"异常"**: 异常样本与正常样本在特征空间中距离不够远
- **训练数据过度平滑**: 过滤异常后的训练数据可能过于"完美"

### 2. 重构任务的固有困难
- **过度泛化**: 模型学会重构所有样本，包括异常样本
- **表示能力过强**: TimesNet的表示能力可能超过了数据的复杂度
- **阈值设定困难**: 重构误差分布重叠严重

### 3. TimesNet架构不匹配
- **周期性假设**: TimesNet假设存在周期性，但Contact数据可能缺乏明显周期
- **时序依赖性**: 异常可能是瞬时的，不依赖长期时序模式

## 💡 解决方案
"""
        
        for i, solution in enumerate(self.results['solutions'], 1):
            report += f"{i}. {solution}\n"
        
        report += """
## 🎯 关键结论

**TimesNet在Contact数据上失败的根本原因可能是:**
1. **数据特性不匹配**: Contact数据缺乏TimesNet擅长的周期性模式
2. **异常定义问题**: 当前的异常标注可能不够明确或一致
3. **重构范式局限**: 对于这类数据，重构可能不是最佳的异常检测方式

**建议采用的替代方案:**
- 基于密度的异常检测 (如Isolation Forest)
- 基于距离的方法 (如LOF)
- 简化的神经网络架构
- 重新审视异常标注标准
"""
        
        with open('reconstruction_failure_analysis.md', 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info("📄 分析报告已保存: reconstruction_failure_analysis.md")
        
    def run_full_analysis(self):
        """运行完整分析"""
        logger.info("🚀 开始重构异常检测失败原因分析...")
        
        # 加载数据
        train_data, test_data, test_labels, scaler = self.load_contact_data()
        
        # 各项分析
        self.analyze_data_characteristics(train_data, test_data, test_labels)
        self.test_simple_reconstruction_baselines(train_data, test_data, test_labels)
        self.analyze_timesnet_specific_issues(train_data, test_data, test_labels)
        
        # 提出解决方案
        self.propose_solutions()
        
        # 生成报告
        self.generate_comprehensive_report()
        
        logger.info("✅ 分析完成！")
        
        return self.results

def main():
    diagnostic = ReconstructionAnomalyDiagnostic()
    results = diagnostic.run_full_analysis()
    return results

if __name__ == "__main__":
    main() 