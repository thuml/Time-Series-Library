"""
数据下采样模块：将按秒采样的数据下采样至分钟级别
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到Python路径
import sys
sys.path.append('/home/wanting/Time-Series-Library')

# 导入自定义logger
from utils.logger import setup_logger

# 创建专用logger
logger = setup_logger("subsample_minute")


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载parquet数据文件
    
    Args:
        file_path: 数据文件路径
    
    Returns:
        加载的DataFrame
    """
    logger.info(f"开始加载数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"数据文件不存在: {file_path}")
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"数据加载成功，形状: {df.shape}")
        logger.info(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        raise


def remove_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    移除指定列
    
    Args:
        df: 输入DataFrame
        columns_to_remove: 要移除的列名列表
    
    Returns:
        移除指定列后的DataFrame
    """
    logger.info(f"移除列: {columns_to_remove}")
    
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"以下列不存在，将被忽略: {missing_columns}")
    
    if existing_columns:
        df_cleaned = df.drop(columns=existing_columns)
        logger.info(f"成功移除列: {existing_columns}")
        logger.info(f"移除后数据形状: {df_cleaned.shape}")
        return df_cleaned
    else:
        logger.warning("没有找到要移除的列")
        return df.copy()


def prepare_timestamp_column(df: pd.DataFrame, timestamp_col: str = 'TimeStamp') -> pd.DataFrame:
    """
    准备时间戳列，确保正确的时间格式并按时间戳排序
    
    Args:
        df: 输入DataFrame
        timestamp_col: 时间戳列名
    
    Returns:
        处理后的DataFrame
    """
    logger.info(f"处理时间戳列: {timestamp_col}")
    
    if timestamp_col not in df.columns:
        logger.error(f"时间戳列 '{timestamp_col}' 不存在")
        raise ValueError(f"时间戳列 '{timestamp_col}' 不存在")
    
    # 确保时间戳是datetime格式
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.info("转换时间戳列为datetime格式")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    logger.info(f"排序前时间戳范围: {df[timestamp_col].min()} 到 {df[timestamp_col].max()}")
    
    # 按时间戳排序
    logger.info("按时间戳排序数据...")
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    logger.info(f"排序后时间戳范围: {df_sorted[timestamp_col].min()} 到 {df_sorted[timestamp_col].max()}")
    logger.info(f"时间戳列数据类型: {df_sorted[timestamp_col].dtype}")
    logger.info(f"数据是否按时间戳排序: {df_sorted[timestamp_col].is_monotonic_increasing}")
    
    return df_sorted


def subsample_to_minute(df: pd.DataFrame, timestamp_col: str = 'TimeStamp', 
                       aggregation_method: str = 'mean', label: str = 'left') -> pd.DataFrame:
    """
    将按秒采样的数据下采样至分钟级别
    
    Args:
        df: 输入DataFrame
        timestamp_col: 时间戳列名
        aggregation_method: 聚合方法 ('mean', 'median', 'first', 'last', 'max', 'min')
        label: 时间戳标签位置 ('left', 'right', 'center')
               - 'left': 使用时间窗口的开始时间 (默认)
               - 'right': 使用时间窗口的结束时间  
               - 'center': 使用时间窗口的中间时间
    
    Returns:
        下采样后的DataFrame
    """
    logger.info(f"开始下采样至分钟级别，聚合方法: {aggregation_method}, 时间戳标签: {label}")
    
    # 设置时间戳为索引
    df_indexed = df.set_index(timestamp_col)
    
    # 获取数值列
    numeric_columns = df_indexed.select_dtypes(include=[np.number]).columns
    logger.info(f"数值列数量: {len(numeric_columns)}")
    
    # 确定时间戳标签参数
    if label == 'left':
        resample_label = 'left'
    elif label == 'right':
        resample_label = 'right'
    elif label == 'center':
        resample_label = 'left'  # 先用left，后面会调整到center
    else:
        logger.warning(f"未知的标签类型 '{label}'，使用默认的'left'")
        resample_label = 'left'
    
    # 下采样到分钟级别
    logger.info("执行下采样操作...")
    
    if aggregation_method == 'mean':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).mean()
    elif aggregation_method == 'median':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).median()
    elif aggregation_method == 'first':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).first()
    elif aggregation_method == 'last':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).last()
    elif aggregation_method == 'max':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).max()
    elif aggregation_method == 'min':
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).min()
    else:
        logger.warning(f"未知的聚合方法 '{aggregation_method}'，使用默认的'mean'")
        df_resampled = df_indexed[numeric_columns].resample('1min', label=resample_label).mean()
    
    # 如果是center标签，需要调整时间戳到中间点
    if label == 'center':
        logger.info("调整时间戳到时间窗口中心点...")
        df_resampled.index = df_resampled.index + pd.Timedelta(seconds=30)
    
    # 重置索引，将时间戳重新设为列
    df_resampled = df_resampled.reset_index()
    
    # 移除包含NaN的行（可能是由于某些分钟没有数据）
    initial_rows = len(df_resampled)
    df_resampled = df_resampled.dropna()
    final_rows = len(df_resampled)
    
    if initial_rows != final_rows:
        logger.info(f"移除了 {initial_rows - final_rows} 行包含NaN的数据")
    
    logger.info(f"下采样完成，新数据形状: {df_resampled.shape}")
    logger.info(f"新时间戳范围: {df_resampled[timestamp_col].min()} 到 {df_resampled[timestamp_col].max()}")
    
    return df_resampled


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    保存处理后的数据到parquet格式
    
    Args:
        df: 要保存的DataFrame
        output_path: 输出文件路径
    """
    logger.info(f"保存处理后的数据到: {output_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"数据保存成功，文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        raise


def process_subsample_minute(input_file: str, 
                           output_file: Optional[str] = None,
                           columns_to_remove: List[str] = ['ID', 'Station'],
                           aggregation_method: str = 'mean',
                           timestamp_col: str = 'TimeStamp',
                           label: str = 'left') -> str:
    """
    完整的下采样处理流程
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选）
        columns_to_remove: 要移除的列名列表
        aggregation_method: 聚合方法
        timestamp_col: 时间戳列名
        label: 时间戳标签位置 ('left', 'right', 'center')
    
    Returns:
        输出文件路径
    """
    logger.info("="*50)
    logger.info("开始数据下采样处理流程")
    logger.info("="*50)
    
    # 步骤1: 加载数据
    df = load_data(input_file)
    
    # 步骤2: 移除指定列
    df_cleaned = remove_columns(df, columns_to_remove)
    
    # 步骤3: 准备时间戳列
    df_prepared = prepare_timestamp_column(df_cleaned, timestamp_col)
    
    # 步骤4: 下采样至分钟级别
    df_subsampled = subsample_to_minute(df_prepared, timestamp_col, aggregation_method, label)
    
    # 步骤5: 生成输出文件路径
    if output_file is None:
        input_path = Path(input_file)
        output_file = f"dataset/data_preprocess/minute_sampled_{input_path.stem}.parquet"
    
    # 步骤6: 保存处理后的数据
    save_processed_data(df_subsampled, output_file)
    
    logger.info("="*50)
    logger.info("数据下采样处理流程完成")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"处理后数据形状: {df_subsampled.shape}")
    logger.info("="*50)
    
    return output_file


def main():
    """
    主函数，执行数据下采样处理
    """
    try:
        # 输入文件路径
        input_file = "dataset/origin_data/cleaning_utc/Contacting_cleaned.parquet"
        
        # 执行下采样处理
        output_file = process_subsample_minute(
            input_file=input_file,
            columns_to_remove=['ID', 'Station'],
            aggregation_method='mean',
            timestamp_col='TimeStamp',
            label='center'
        )
        
        logger.info(f"处理完成！输出文件: {output_file}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
