#!/usr/bin/env python
"""
Contact数据集最终分析脚本
整合了之前版本的优点，提供可靠的分析结果
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def analyze_contact_data():
    """
    Contact数据集的确定性分析
    """
    print("🔍 Contact数据集 - 最终分析报告")
    print("="*60)
    
    # 1. 加载数据
    try:
        train_df = pd.read_parquet("./dataset/anomaly_filtered_Traindata/train_normal.parquet")
        test_df = pd.read_parquet("./dataset/split_data/test.parquet")
        test_labels = pd.read_parquet("./dataset/anomaly_labeld_testData/test_labeled.parquet")
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 数据基本信息
    print(f"\n📊 数据基本信息:")
    print(f"   训练数据: {train_df.shape}")
    print(f"   测试数据: {test_df.shape}")
    print(f"   测试标签: {test_labels.shape}")
    
    # 3. 获取有效特征列
    exclude_cols = ['TimeStamp', 'timestamp', 'index', 'Unnamed: 0']
    numeric_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    print(f"   有效特征数: {len(numeric_cols)}")
    
    # 4. 数据质量检查
    train_nans = train_df[numeric_cols].isnull().sum().sum()
    test_nans = test_df[numeric_cols].isnull().sum().sum()
    print(f"   训练数据NaN: {train_nans}")
    print(f"   测试数据NaN: {test_nans}")
    
    # 6. 异常分布分析
    normal_mask = test_labels['label'] == 0
    anomaly_mask = test_labels['label'] == 1
    
    print(f"\n🎯 异常分布:")
    print(f"   正常样本: {normal_mask.sum()} ({normal_mask.sum()/len(test_labels)*100:.1f}%)")
    print(f"   异常样本: {anomaly_mask.sum()} ({anomaly_mask.sum()/len(test_labels)*100:.1f}%)")
    
    # 7. 特征区分力分析
    print(f"\n🔍 特征区分力分析:")
    feature_scores = []
    
    for col in numeric_cols:
        normal_data = test_df[normal_mask][col]
        anomaly_data = test_df[anomaly_mask][col]
        
        # 标准化均值差异
        mean_diff = abs(normal_data.mean() - anomaly_data.mean())
        std_pooled = np.sqrt((normal_data.std()**2 + anomaly_data.std()**2) / 2) + 1e-8
        
        discrimination_score = mean_diff / std_pooled
        feature_scores.append((col, discrimination_score))
    
    # 排序特征
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Top 5 最有区分力的特征:")
    for i, (feature, score) in enumerate(feature_scores[:5]):
        print(f"     {i+1}. {feature}: {score:.4f}")
    
    avg_score = np.mean([score for _, score in feature_scores])
    print(f"   平均区分力: {avg_score:.4f}")
    
    # 8. 相关性分析
    print(f"\n🔗 相关性分析:")
    corr_matrix = train_df[numeric_cols].corr()
    high_corr_count = 0
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_count += 1
    
    print(f"   高相关性特征对 (>0.9): {high_corr_count}")
    
    # 9. PCA分析
    print(f"\n🧮 PCA分析:")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[numeric_cols])
    test_scaled = scaler.transform(test_df[numeric_cols])
    
    n_components = min(10, len(numeric_cols))
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    print(f"   前5个主成分解释方差: {pca.explained_variance_ratio_[:5]}")
    print(f"   累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 10. 时序模式分析（使用最佳特征）
    print(f"\n⏰ 时序模式分析:")
    best_feature = feature_scores[0][0]  # 最有区分力的特征
    print(f"   分析特征: {best_feature}")
    
    # 采样数据进行自相关分析
    sample_size = min(1000, normal_mask.sum())
    normal_sample = test_df[normal_mask][best_feature].iloc[:sample_size]
    anomaly_sample = test_df[anomaly_mask][best_feature].iloc[:min(1000, anomaly_mask.sum())]
    
    def autocorr(x, max_lag=50):
        """计算自相关"""
        x = x - x.mean()  # 去中心化
        result = np.correlate(x, x, mode='full')
        result = result[len(result)//2:]
        if result[0] != 0:
            result = result / result[0]
        return result[:max_lag]
    
    normal_autocorr = autocorr(normal_sample.values)
    anomaly_autocorr = autocorr(anomaly_sample.values)
    
    autocorr_diff = np.mean(np.abs(normal_autocorr - anomaly_autocorr))
    print(f"   时序模式差异: {autocorr_diff:.4f}")
    
    # 11. 最终建议
    print(f"\n🎯 最终建议:")
    
    if avg_score < 0.1:
        difficulty = "困难"
        seq_len_rec = "300-500"
        epochs_rec = "25-30"
        model_rec = "深度模型 (e_layers>=6)"
    elif avg_score < 0.2:
        difficulty = "中等"
        seq_len_rec = "200-300"
        epochs_rec = "20-25"
        model_rec = "标准模型 (e_layers=4-6)"
    else:
        difficulty = "相对容易"
        seq_len_rec = "100-200"
        epochs_rec = "15-20"
        model_rec = "标准模型即可"
    
    print(f"   异常检测难度: {difficulty}")
    print(f"   推荐序列长度: {seq_len_rec}")
    print(f"   推荐训练轮数: {epochs_rec}")
    print(f"   推荐模型配置: {model_rec}")
    
    if autocorr_diff < 0.1:
        print(f"   ⚠️  时序模式差异较小，建议增强特征工程")
    else:
        print(f"   ✅ 时序模式有差异，模型应该能够学习")
    
    if high_corr_count > 20:
        print(f"   ⚠️  特征冗余较多，建议特征选择")
    
    print(f"\n✅ 分析完成！基于这个分析结果进行模型训练。")
    
    # 返回关键指标
    return {
        'avg_discrimination': avg_score,
        'temporal_difference': autocorr_diff,
        'high_correlation_pairs': high_corr_count,
        'best_feature': best_feature,
        'anomaly_ratio': anomaly_mask.sum() / len(test_labels),
        'difficulty': difficulty
    }

if __name__ == "__main__":
    results = analyze_contact_data() 