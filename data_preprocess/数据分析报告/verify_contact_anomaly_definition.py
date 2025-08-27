import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from utils.logger import logger

def verify_contact_anomaly_definition():
    """验证Contact数据集的异常定义"""
    logger.info("验证Contact异常定义...")
    
    # 1. 加载数据
    contact_test = pd.read_parquet("./dataset/split_data/test.parquet")
    contact_labels = pd.read_parquet("./dataset/anomaly_labeld_testData/test_labeled.parquet")
    
    # 跳过TimeStamp列
    test_data = contact_test.iloc[:, 1:].values
    test_labels = contact_labels.iloc[:, 1:].values.flatten()
    
    logger.info(f"测试数据形状: {test_data.shape}")
    logger.info(f"异常样本数: {test_labels.sum()} / {len(test_labels)} ({test_labels.sum()/len(test_labels)*100:.2f}%)")
    
    # 2. 标准化数据
    scaler = StandardScaler()
    test_data_scaled = scaler.fit_transform(test_data)
    
    # 3. 分析正常vs异常样本的统计特性
    normal_mask = test_labels == 0
    anomaly_mask = test_labels == 1
    
    normal_data = test_data_scaled[normal_mask]
    anomaly_data = test_data_scaled[anomaly_mask]
    
    logger.info(f"\n=== 统计特性对比 ===")
    logger.info(f"正常样本统计:")
    logger.info(f"  数量: {len(normal_data)}")
    logger.info(f"  均值范围: [{normal_data.mean(axis=0).min():.4f}, {normal_data.mean(axis=0).max():.4f}]")
    logger.info(f"  标准差范围: [{normal_data.std(axis=0).min():.4f}, {normal_data.std(axis=0).max():.4f}]")
    logger.info(f"  方差总和: {normal_data.var().sum():.4f}")
    
    logger.info(f"\n异常样本统计:")
    logger.info(f"  数量: {len(anomaly_data)}")
    logger.info(f"  均值范围: [{anomaly_data.mean(axis=0).min():.4f}, {anomaly_data.mean(axis=0).max():.4f}]")
    logger.info(f"  标准差范围: [{anomaly_data.std(axis=0).min():.4f}, {anomaly_data.std(axis=0).max():.4f}]")
    logger.info(f"  方差总和: {anomaly_data.var().sum():.4f}")
    
    # 4. 使用无监督异常检测验证
    logger.info(f"\n=== 无监督异常检测验证 ===")
    
    # 4.1 Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_predictions = iso_forest.fit_predict(test_data_scaled)
    iso_anomalies = (iso_predictions == -1)
    
    # 计算与真实标签的一致性
    iso_agreement = (iso_anomalies == (test_labels == 1)).mean()
    logger.info(f"Isolation Forest一致性: {iso_agreement:.4f}")
    
    # 4.2 PCA重构误差
    pca = PCA(n_components=10)
    pca.fit(test_data_scaled)
    
    reconstructed = pca.inverse_transform(pca.transform(test_data_scaled))
    reconstruction_errors = np.mean((test_data_scaled - reconstructed) ** 2, axis=1)
    
    # 按重构误差排序，前5%作为异常
    error_threshold = np.percentile(reconstruction_errors, 95)
    pca_anomalies = reconstruction_errors > error_threshold
    
    pca_agreement = (pca_anomalies == (test_labels == 1)).mean()
    logger.info(f"PCA重构误差一致性: {pca_agreement:.4f}")
    
    # 5. 分析错误分类的样本
    logger.info(f"\n=== 错误分类分析 ===")
    
    # 被标记为正常但重构误差高的样本
    false_normal = normal_mask & (reconstruction_errors > np.median(reconstruction_errors[anomaly_mask]))
    false_normal_count = false_normal.sum()
    
    # 被标记为异常但重构误差低的样本  
    false_anomaly = anomaly_mask & (reconstruction_errors < np.median(reconstruction_errors[normal_mask]))
    false_anomaly_count = false_anomaly.sum()
    
    logger.info(f"被标记为正常但重构误差高的样本: {false_normal_count}")
    logger.info(f"被标记为异常但重构误差低的样本: {false_anomaly_count}")
    
    # 6. 特征级别的异常分析
    logger.info(f"\n=== 特征级异常分析 ===")
    feature_names = list(contact_test.columns[1:])
    
    suspicious_features = []
    for i, feature_name in enumerate(feature_names):
        normal_vals = test_data_scaled[normal_mask, i]
        anomaly_vals = test_data_scaled[anomaly_mask, i]
        
        # 检查异常样本是否更接近0（标准化后）
        normal_abs_mean = np.abs(normal_vals).mean()
        anomaly_abs_mean = np.abs(anomaly_vals).mean()
        
        if anomaly_abs_mean < normal_abs_mean:
            ratio = normal_abs_mean / (anomaly_abs_mean + 1e-8)
            suspicious_features.append((feature_name, ratio))
    
    suspicious_features.sort(key=lambda x: x[1], reverse=True)
    
    if suspicious_features:
        logger.warning(f"发现可疑特征（异常样本更接近均值）:")
        for name, ratio in suspicious_features[:5]:
            logger.warning(f"  {name}: 比值={ratio:.2f}")
    
    # 7. 时间序列模式分析
    logger.info(f"\n=== 时序模式分析 ===")
    
    # 分析前1000个样本的时序特性
    sample_size = min(1000, len(test_data))
    sample_data = test_data_scaled[:sample_size]
    sample_labels = test_labels[:sample_size]
    
    normal_indices = np.where(sample_labels == 0)[0]
    anomaly_indices = np.where(sample_labels == 1)[0]
    
    if len(normal_indices) > 0 and len(anomaly_indices) > 0:
        # 计算相邻时间点的变化率
        normal_changes = []
        anomaly_changes = []
        
        for i in normal_indices[:-1]:
            if i+1 in normal_indices:
                change = np.linalg.norm(sample_data[i+1] - sample_data[i])
                normal_changes.append(change)
        
        for i in anomaly_indices[:-1]:
            if i+1 in anomaly_indices:
                change = np.linalg.norm(sample_data[i+1] - sample_data[i])
                anomaly_changes.append(change)
        
        if normal_changes and anomaly_changes:
            logger.info(f"正常样本平均变化率: {np.mean(normal_changes):.4f}")
            logger.info(f"异常样本平均变化率: {np.mean(anomaly_changes):.4f}")
    
    # 8. 结论和建议
    logger.info(f"\n=== 结论 ===")
    
    if pca_agreement < 0.3:
        logger.error("❌ PCA重构误差与标签严重不一致！")
        logger.error("可能原因：")
        logger.error("  1. 异常标签定义与重构复杂度无关")
        logger.error("  2. 数据预处理丢失了关键信息")
        logger.error("  3. 异常可能是基于其他维度定义的（如时间、业务逻辑）")
    
    if iso_agreement < 0.3:
        logger.error("❌ Isolation Forest与标签也不一致！")
        logger.error("这进一步确认了标签定义的问题")
    
    logger.info(f"\n建议：")
    logger.info(f"  1. 重新审视异常的定义和标注过程")
    logger.info(f"  2. 考虑使用监督学习方法而非重构方法")
    logger.info(f"  3. 分析原始时间序列数据而非静态特征")
    logger.info(f"  4. 与数据提供方确认异常的具体含义")

if __name__ == "__main__":
    verify_contact_anomaly_definition() 