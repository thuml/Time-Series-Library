#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import os
import numpy as np

def label_anomalies(test_path, anomaly_dict_path, station_name="Kontaktieren", output_dir="dataset/anomaly_labeld_testData"):
    """
    Label test data points as anomalies (1) or normal (0) based on anomaly time ranges.
    
    Args:
        test_path (str): Path to the test Parquet file
        anomaly_dict_path (str): Path to the anomaly dictionary pickle file
        station_name (str): Name of the station to get anomalies for
        output_dir (str): Directory to save the labeled dataset
    """
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_parquet(test_path)
    print(f"Test data loaded. Total rows: {len(test_df)}")
    
    print(f"Loading anomaly dictionary from: {anomaly_dict_path}")
    with open(anomaly_dict_path, 'rb') as f:
        anomaly_dict = pickle.load(f)
    
    # Check if the station exists in the anomaly dictionary
    if station_name not in anomaly_dict:
        print(f"Station '{station_name}' not found in anomaly dictionary. Available stations: {list(anomaly_dict.keys())}")
        return
    
    station_anomalies = anomaly_dict[station_name]
    print(f"Found {len(station_anomalies)} anomaly time ranges for station '{station_name}'")
    
    # Convert timestamp column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(test_df['TimeStamp']):
        test_df['TimeStamp'] = pd.to_datetime(test_df['TimeStamp'])
    
    # Initialize label column with 0 (normal)
    test_df['label'] = 0
    
    # Create a copy of the DataFrame with only TimeStamp and label columns
    labeled_df = test_df[['TimeStamp', 'label']].copy()
    
    print("Labeling anomalies in test data...")
    
    # Count anomalies for reporting
    anomaly_count = 0
    
    # Check each anomaly time range
    for anomaly_period in station_anomalies:
        # Handle tuple structure (start time, end time)
        if isinstance(anomaly_period, tuple) and len(anomaly_period) >= 2:
            start_time = pd.to_datetime(anomaly_period[0])
            end_time = pd.to_datetime(anomaly_period[1])
        else:
            print(f"Unexpected anomaly format: {anomaly_period}, skipping...")
            continue
        
        # Find rows that fall within this anomaly period and label them as 1 (anomaly)
        anomaly_mask = (labeled_df['TimeStamp'] >= start_time) & (labeled_df['TimeStamp'] <= end_time)
        labeled_df.loc[anomaly_mask, 'label'] = 1
        
        # Count newly identified anomalies
        anomaly_count += anomaly_mask.sum()
    
    print(f"Labeling complete. Found {anomaly_count} anomaly data points out of {len(labeled_df)} total.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the labeled dataset
    output_path = os.path.join(output_dir, "test_labeled.parquet")
    print(f"Saving labeled dataset to: {output_path}")
    labeled_df.to_parquet(output_path, index=False)
    
    # Optional: Save a CSV version for easier inspection
    csv_output_path = os.path.join(output_dir, "test_labeled.csv")
    labeled_df.to_csv(csv_output_path, index=False)
    print(f"Also saved as CSV for easier inspection: {csv_output_path}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total data points: {len(labeled_df)}")
    print(f"Normal data points (label=0): {(labeled_df['label'] == 0).sum()}")
    print(f"Anomaly data points (label=1): {(labeled_df['label'] == 1).sum()}")
    print(f"Anomaly percentage: {(labeled_df['label'] == 1).sum() / len(labeled_df) * 100:.2f}%")
    
    print("Labeling process complete!")
    
    return labeled_df

if __name__ == "__main__":
    test_path = "dataset/split_data/test.parquet"
    anomaly_dict_path = "dataset/origin_data/anomaly_dict_merged.pkl"
    
    label_anomalies(test_path, anomaly_dict_path) 