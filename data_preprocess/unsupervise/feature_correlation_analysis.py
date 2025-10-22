#!/usr/bin/env python3
"""
特征相关性分析和冗余特征删除
"""

import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Set
from utils.logger import setup_logger
import warnings

warnings.filterwarnings('ignore')

def load_training_data(file_path: str) -> pd.DataFrame:
    """
    加载训练数据
    
    Args:
        file_path: 训练数据文件路径
        
    Returns:
        pd.DataFrame: 加载的训练数据
    """
    logger = setup_logger("load_training_data")
    
    try:
        logger.info(f"Loading training data from: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded training data. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise


def calculate_correlation_matrix(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    计算特征相关性矩阵
    
    Args:
        df: 输入数据框
        exclude_cols: 需要排除的列名列表（如时间戳、标签等）
        
    Returns:
        pd.DataFrame: 相关性矩阵
    """
    logger = setup_logger("calculate_correlation_matrix")
    
    if exclude_cols is None:
        exclude_cols = []
    
    try:
        # 选择数值型列进行相关性分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除指定的列
        analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Calculating correlation matrix for {len(analysis_cols)} features")
        logger.info(f"Excluded columns: {exclude_cols}")
        
        # 计算相关性矩阵
        correlation_matrix = df[analysis_cols].corr()
        
        logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
        
        # 保存相关性矩阵
        output_dir = "experiments/intermediate_results"
        os.makedirs(output_dir, exist_ok=True)
        correlation_file = os.path.join(output_dir, "correlation_matrix.parquet")
        correlation_matrix.to_parquet(correlation_file)
        logger.info(f"Correlation matrix saved to: {correlation_file}")
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {str(e)}")
        raise


def find_highly_correlated_features(correlation_matrix: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    找出高度相关的特征对
    
    Args:
        correlation_matrix: 相关性矩阵
        threshold: 相关性阈值
        
    Returns:
        List[Tuple[str, str, float]]: 高度相关的特征对列表 (feature1, feature2, correlation)
    """
    logger = setup_logger("find_highly_correlated_features")
    
    try:
        logger.info(f"Finding highly correlated features with threshold: {threshold}")
        
        highly_correlated_pairs = []
        
        # 遍历相关性矩阵的上三角部分
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                # 检查是否超过阈值
                if abs(correlation) > threshold and not pd.isna(correlation):
                    highly_correlated_pairs.append((feature1, feature2, correlation))
        
        logger.info(f"Found {len(highly_correlated_pairs)} highly correlated feature pairs")
        
        # 按相关性大小排序
        highly_correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 记录前10个最相关的特征对
        logger.info("Top 10 highly correlated feature pairs:")
        for i, (feat1, feat2, corr) in enumerate(highly_correlated_pairs[:10]):
            logger.info(f"  {i+1}. {feat1} <-> {feat2}: {corr:.4f}")
        
        return highly_correlated_pairs
        
    except Exception as e:
        logger.error(f"Error finding highly correlated features: {str(e)}")
        raise


def select_features_to_remove(highly_correlated_pairs: List[Tuple[str, str, float]], 
                            df: pd.DataFrame) -> Set[str]:
    """
    选择需要删除的冗余特征
    
    Args:
        highly_correlated_pairs: 高度相关的特征对列表
        df: 原始数据框（用于计算特征的方差等统计信息）
        
    Returns:
        Set[str]: 需要删除的特征名集合
    """
    logger = setup_logger("select_features_to_remove")
    
    try:
        logger.info("Selecting redundant features to remove")
        
        features_to_remove = set()
        
        # 计算每个特征的方差（用于决定保留哪个特征）
        feature_variance = df.select_dtypes(include=[np.number]).var()
        
        for feature1, feature2, correlation in highly_correlated_pairs:
            # 如果两个特征都还没有被标记为删除
            if feature1 not in features_to_remove and feature2 not in features_to_remove:
                # 保留方差更大的特征（信息量更丰富）
                if feature_variance[feature1] >= feature_variance[feature2]:
                    features_to_remove.add(feature2)
                    logger.info(f"Marked {feature2} for removal (corr with {feature1}: {correlation:.4f})")
                else:
                    features_to_remove.add(feature1)
                    logger.info(f"Marked {feature1} for removal (corr with {feature2}: {correlation:.4f})")
        
        logger.info(f"Total features to remove: {len(features_to_remove)}")
        logger.info(f"Features to remove: {sorted(list(features_to_remove))}")
        
        # 保存要删除的特征列表
        output_dir = "experiments/intermediate_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为文本文件便于查看
        features_file = os.path.join(output_dir, "features_to_remove.txt")
        with open(features_file, 'w') as f:
            for feature in sorted(features_to_remove):
                f.write(f"{feature}\n")
        logger.info(f"Features to remove saved to: {features_file}")
        
        return features_to_remove
        
    except Exception as e:
        logger.error(f"Error selecting features to remove: {str(e)}")
        raise


def remove_redundant_features(df: pd.DataFrame, features_to_remove: Set[str]) -> pd.DataFrame:
    """
    从数据框中删除冗余特征
    
    Args:
        df: 原始数据框
        features_to_remove: 需要删除的特征名集合
        
    Returns:
        pd.DataFrame: 删除冗余特征后的数据框
    """
    logger = setup_logger("remove_redundant_features")
    
    try:
        original_shape = df.shape
        logger.info(f"Original data shape: {original_shape}")
        
        # 删除冗余特征
        df_filtered = df.drop(columns=list(features_to_remove), errors='ignore')
        
        new_shape = df_filtered.shape
        logger.info(f"Filtered data shape: {new_shape}")
        logger.info(f"Removed {original_shape[1] - new_shape[1]} features")
        
        return df_filtered
        
    except Exception as e:
        logger.error(f"Error removing redundant features: {str(e)}")
        raise


def save_filtered_data(df: pd.DataFrame, output_path: str):
    """
    保存过滤后的数据
    
    Args:
        df: 过滤后的数据框
        output_path: 输出文件路径
    """
    logger = setup_logger("save_filtered_data")
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Saving filtered data to: {output_path}")
        df.to_parquet(output_path, index=False)
        
        file_size = os.path.getsize(output_path) / 1024**2
        logger.info(f"Successfully saved filtered data. File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error saving filtered data: {str(e)}")
        raise


def process_feature_correlation_analysis(train_data_path: str = "dataset/anomaly_filtered_Traindata/train_normal.parquet",
                                       test_data_path: str = "dataset/split_data/test.parquet",
                                       correlation_threshold: float = 0.95,
                                       exclude_cols: List[str] = None):
    """
    主函数：执行完整的特征相关性分析和冗余特征删除流程
    
    Args:
        train_data_path: 训练数据路径
        test_data_path: 测试数据路径
        correlation_threshold: 相关性阈值
        exclude_cols: 需要排除的列名列表
    """
    main_logger = setup_logger("feature_correlation_analysis")
    
    try:
        main_logger.info("=== Starting Feature Correlation Analysis ===")
        main_logger.info(f"Train data path: {train_data_path}")
        main_logger.info(f"Test data path: {test_data_path}")
        main_logger.info(f"Correlation threshold: {correlation_threshold}")
        
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'date', 'label', 'anomaly']  # 常见的非特征列
        
        # 步骤1: 加载训练数据
        main_logger.info("Step 1: Loading training data")
        train_df = load_training_data(train_data_path)
        
        # 步骤2: 计算相关性矩阵
        main_logger.info("Step 2: Calculating correlation matrix")
        correlation_matrix = calculate_correlation_matrix(train_df, exclude_cols)
        
        # 步骤3: 找出高度相关的特征对
        main_logger.info("Step 3: Finding highly correlated features")
        highly_correlated_pairs = find_highly_correlated_features(correlation_matrix, correlation_threshold)
        
        # 步骤4: 选择需要删除的特征
        main_logger.info("Step 4: Selecting features to remove")
        features_to_remove = select_features_to_remove(highly_correlated_pairs, train_df)
        
        if not features_to_remove:
            main_logger.info("No highly correlated features found. No features to remove.")
            return
        
        # 步骤5: 从训练数据中删除冗余特征
        main_logger.info("Step 5: Removing redundant features from training data")
        train_df_filtered = remove_redundant_features(train_df, features_to_remove)
        
        # 保存过滤后的训练数据
        output_dir = "dataset/feature_filtered_data"
        train_output_path = os.path.join(output_dir, "train_normal_filtered.parquet")
        save_filtered_data(train_df_filtered, train_output_path)
        
        # 步骤6: 处理测试数据
        main_logger.info("Step 6: Processing test data")
        if os.path.exists(test_data_path):
            main_logger.info(f"Loading test data from: {test_data_path}")
            test_df = pd.read_parquet(test_data_path)
            main_logger.info(f"Test data shape: {test_df.shape}")
            
            # 从测试数据中删除相同的特征
            test_df_filtered = remove_redundant_features(test_df, features_to_remove)
            
            # 保存过滤后的测试数据
            test_output_path = os.path.join(output_dir, "test_filtered.parquet")
            save_filtered_data(test_df_filtered, test_output_path)
        else:
            main_logger.warning(f"Test data file not found: {test_data_path}")
        
        main_logger.info("=== Feature Correlation Analysis Completed Successfully ===")
        main_logger.info(f"Filtered data saved to: {output_dir}")
        
    except Exception as e:
        main_logger.error(f"Error in feature correlation analysis: {str(e)}")
        raise


if __name__ == "__main__":
    # 执行特征相关性分析
    process_feature_correlation_analysis() 