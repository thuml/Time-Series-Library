#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def preview_parquet_file(file_path, n_rows=5):
    """
    Preview a Parquet file by showing first and last n rows
    
    Args:
        file_path (str): Path to the Parquet file
        n_rows (int): Number of rows to display from beginning and end
    """
    try:
        print(f"Loading Parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        print(f"File loaded successfully. Total rows: {len(df)}, columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print(f"\n--- First {n_rows} rows ---")
        print(df.head(n_rows))
        
        print(f"\n--- Last {n_rows} rows ---")
        print(df.tail(n_rows))
      
    except Exception as e:
        print(f"Error previewing file {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Hardcoded file path
    file_path = "dataset/supervised/lable/train_labeled.parquet"
    preview_parquet_file(file_path, 5) 