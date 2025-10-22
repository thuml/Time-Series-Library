import pandas as pd
import pickle
import numpy as np
from utils.logger import logger
from tqdm import tqdm


def label_anomaly_data(train_path: str = "dataset/split_data/train.parquet",
                      test_path: str = "dataset/split_data/test.parquet", 
                      anomaly_dict_path: str = "dataset/origin_data/anomaly_dict_merged.pkl",
                      station_key: str = "Kontaktieren",
                      output_train_path: str = "dataset/split_data/train_labeled.parquet",
                      output_test_path: str = "dataset/split_data/test_labeled.parquet"):
    """
    对时间序列数据进行异常标签操作
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        anomaly_dict_path: 异常字典pkl文件路径
        station_key: 站点名称键值 (默认: "Kontaktieren")
        output_train_path: 输出训练数据路径
        output_test_path: 输出测试数据路径
    
    Returns:
        tuple: (labeled_train_df, labeled_test_df) 标记后的训练和测试数据
    """
    
    logger.info("开始加载数据文件...")
    
    # 加载数据
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        logger.info(f"训练数据形状: {train_df.shape}")
        logger.info(f"测试数据形状: {test_df.shape}")
        
        with open(anomaly_dict_path, 'rb') as f:
            anomaly_dict = pickle.load(f)
        logger.info(f"异常字典包含的站点: {list(anomaly_dict.keys())}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    # 检查站点键是否存在
    if station_key not in anomaly_dict:
        available_keys = list(anomaly_dict.keys())
        logger.error(f"站点键 '{station_key}' 不在异常字典中。可用键: {available_keys}")
        raise ValueError(f"站点键 '{station_key}' 不存在")
    
    # 获取异常时间段
    anomaly_periods = anomaly_dict[station_key]
    logger.info(f"站点 '{station_key}' 共有 {len(anomaly_periods)} 个异常时间段")
    
    # 为训练数据添加标签
    logger.info("开始为训练数据添加标签...")
    train_df_labeled = _add_anomaly_labels(train_df, anomaly_periods, "训练")
    
    # 为测试数据添加标签
    logger.info("开始为测试数据添加标签...")
    test_df_labeled = _add_anomaly_labels(test_df, anomaly_periods, "测试")
    
    # 保存标记后的数据
    logger.info("保存标记后的数据...")
    train_df_labeled.to_parquet(output_train_path, index=False)
    test_df_labeled.to_parquet(output_test_path, index=False)
    
    # 输出统计信息
    _print_label_statistics(train_df_labeled, test_df_labeled)
    
    logger.info(f"标记完成！训练数据保存至: {output_train_path}")
    logger.info(f"标记完成！测试数据保存至: {output_test_path}")
    
    return train_df_labeled, test_df_labeled


def _add_anomaly_labels(df: pd.DataFrame, anomaly_periods: list, dataset_name: str) -> pd.DataFrame:
    """
    为数据添加异常标签
    
    Args:
        df: 输入数据框
        anomaly_periods: 异常时间段列表，每个元素为(start_time, end_time)的元组
        dataset_name: 数据集名称，用于日志显示
    
    Returns:
        pd.DataFrame: 添加了label列的数据框
    """
    
    # 复制数据框
    df_labeled = df.copy()
    
    # 初始化所有标签为0（正常）
    df_labeled['label'] = 0
    
    # 确保TimeStamp列是datetime类型且带时区信息
    if not pd.api.types.is_datetime64_any_dtype(df_labeled['TimeStamp']):
        df_labeled['TimeStamp'] = pd.to_datetime(df_labeled['TimeStamp'])
    
    # 如果TimeStamp没有时区信息，假设为UTC
    if df_labeled['TimeStamp'].dt.tz is None:
        df_labeled['TimeStamp'] = df_labeled['TimeStamp'].dt.tz_localize('UTC')
    
    anomaly_count = 0
    total_anomaly_periods = len(anomaly_periods)
    
    logger.info(f"开始处理 {dataset_name} 数据的 {total_anomaly_periods} 个异常时间段...")
    
    # 遍历所有异常时间段
    for i, (start_time, end_time) in enumerate(tqdm(anomaly_periods, desc=f"处理{dataset_name}数据异常时间段")):
        # 确保异常时间段的时间戳也有时区信息
        if hasattr(start_time, 'tz') and start_time.tz is None:
            start_time = start_time.tz_localize('UTC')
        if hasattr(end_time, 'tz') and end_time.tz is None:
            end_time = end_time.tz_localize('UTC')
        
        # 找到在异常时间段内的数据点
        mask = (df_labeled['TimeStamp'] >= start_time) & (df_labeled['TimeStamp'] <= end_time)
        anomaly_points = mask.sum()
        
        if anomaly_points > 0:
            df_labeled.loc[mask, 'label'] = 1
            anomaly_count += anomaly_points
            
        # 每处理100个时间段输出一次进度
        if (i + 1) % 100 == 0:
            logger.info(f"已处理 {i + 1}/{total_anomaly_periods} 个异常时间段")
    
    logger.info(f"{dataset_name} 数据标记完成，共标记 {anomaly_count} 个异常点")
    
    return df_labeled


def _print_label_statistics(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    打印标签统计信息
    
    Args:
        train_df: 训练数据框
        test_df: 测试数据框
    """
    
    logger.info("=" * 50)
    logger.info("标签统计信息:")
    logger.info("=" * 50)
    
    # 训练数据统计
    train_normal = (train_df['label'] == 0).sum()
    train_anomaly = (train_df['label'] == 1).sum()
    train_total = len(train_df)
    train_anomaly_ratio = train_anomaly / train_total * 100
    
    logger.info(f"训练数据:")
    logger.info(f"  总样本数: {train_total:,}")
    logger.info(f"  正常样本: {train_normal:,} ({100-train_anomaly_ratio:.2f}%)")
    logger.info(f"  异常样本: {train_anomaly:,} ({train_anomaly_ratio:.2f}%)")
    
    # 测试数据统计
    test_normal = (test_df['label'] == 0).sum()
    test_anomaly = (test_df['label'] == 1).sum()
    test_total = len(test_df)
    test_anomaly_ratio = test_anomaly / test_total * 100
    
    logger.info(f"测试数据:")
    logger.info(f"  总样本数: {test_total:,}")
    logger.info(f"  正常样本: {test_normal:,} ({100-test_anomaly_ratio:.2f}%)")
    logger.info(f"  异常样本: {test_anomaly:,} ({test_anomaly_ratio:.2f}%)")
    
    # 总体统计
    total_samples = train_total + test_total
    total_anomalies = train_anomaly + test_anomaly
    total_anomaly_ratio = total_anomalies / total_samples * 100
    
    logger.info(f"总体统计:")
    logger.info(f"  总样本数: {total_samples:,}")
    logger.info(f"  异常样本: {total_anomalies:,} ({total_anomaly_ratio:.2f}%)")
    logger.info("=" * 50)


def verify_labeled_data(train_path: str = "dataset/split_data/train_labeled.parquet",
                       test_path: str = "dataset/split_data/test_labeled.parquet"):
    """
    验证标记后的数据
    
    Args:
        train_path: 标记后的训练数据路径
        test_path: 标记后的测试数据路径
    """
    
    logger.info("开始验证标记后的数据...")
    
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # 检查是否有label列
        if 'label' not in train_df.columns:
            logger.error("训练数据中没有找到label列")
            return False
        if 'label' not in test_df.columns:
            logger.error("测试数据中没有找到label列")
            return False
        
        # 检查label值是否只包含0和1
        train_unique_labels = set(train_df['label'].unique())
        test_unique_labels = set(test_df['label'].unique())
        
        expected_labels = {0, 1}
        if not train_unique_labels.issubset(expected_labels):
            logger.error(f"训练数据label值异常: {train_unique_labels}")
            return False
        if not test_unique_labels.issubset(expected_labels):
            logger.error(f"测试数据label值异常: {test_unique_labels}")
            return False
        
        # 打印验证统计
        _print_label_statistics(train_df, test_df)
        
        logger.info("数据验证通过！")
        return True
        
    except Exception as e:
        logger.error(f"数据验证失败: {e}")
        return False


if __name__ == "__main__":
    # 示例用法
    try:
        # 执行标签操作
        train_labeled, test_labeled = label_anomaly_data()
        
        # 验证结果
        verify_labeled_data()
        
        logger.info("异常标签操作成功完成！")
        
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise 