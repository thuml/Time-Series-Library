#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

def split_time_series_data(file_path, output_dir='dataset/split_data', time_col='TimeStamp', train_size=0.75):
    """
    Split time series data into train and test sets based on time sorting.
    
    Args:
        file_path (str): Path to the input Parquet file
        output_dir (str): Directory to save the output files
        time_col (str): Name of the timestamp column
        train_size (float): Proportion of data to use for training
    """
    print(f"Loading data from: {file_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read parquet file
    df = pd.read_parquet(file_path)
    print(f"Data loaded. Total rows: {len(df)}")
    
    # Sort by timestamp
    print(f"Sorting data by {time_col}...")
    df = df.sort_values(by=time_col)
    
    # Calculate split point
    split_idx = int(len(df) * train_size)
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data split: train={len(train_df)} rows, test={len(test_df)} rows")
    
    # Get file name without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save train and test data
    train_path = os.path.join(output_dir, f"train.parquet")
    test_path = os.path.join(output_dir, f"test.parquet")
    
    print(f"Saving train data to: {train_path}")
    train_df.to_parquet(train_path, index=False)
    
    print(f"Saving test data to: {test_path}")
    test_df.to_parquet(test_path, index=False)
    
    # Print time range for each set
    print("\nTime ranges:")
    print(f"Train: {train_df[time_col].min()} to {train_df[time_col].max()}")
    print(f"Test: {test_df[time_col].min()} to {test_df[time_col].max()}")
    
    print("\nData split and saved successfully!")

if __name__ == "__main__":
    # Input file path
    file_path = "dataset/data_preprocess/minute_sampled_Contacting_cleaned.parquet"
    
    # Split data with default parameters
    split_time_series_data(file_path) 