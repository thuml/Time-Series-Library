#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import os
from datetime import datetime
import numpy as np

def filter_anomalies(train_path, anomaly_dict_path, station_name="Kontaktieren", output_dir="dataset/anomaly_filtered_data"):
    """
    Filter out anomaly data points from the training dataset based on anomaly time ranges.
    
    Args:
        train_path (str): Path to the training Parquet file
        anomaly_dict_path (str): Path to the anomaly dictionary pickle file
        station_name (str): Name of the station to filter anomalies for
        output_dir (str): Directory to save the filtered dataset
    """
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"Training data loaded. Total rows: {len(train_df)}")
    
    print(f"Loading anomaly dictionary from: {anomaly_dict_path}")
    with open(anomaly_dict_path, 'rb') as f:
        anomaly_dict = pickle.load(f)
    
    # Print structure of the first anomaly to debug
    print(f"Structure of anomaly_dict: {type(anomaly_dict)}")
    if station_name in anomaly_dict and len(anomaly_dict[station_name]) > 0:
        print(f"First anomaly structure: {type(anomaly_dict[station_name][0])}")
        print(f"Example anomaly: {anomaly_dict[station_name][0]}")
    
    # Check if the station exists in the anomaly dictionary
    if station_name not in anomaly_dict:
        print(f"Station '{station_name}' not found in anomaly dictionary. Available stations: {list(anomaly_dict.keys())}")
        return
    
    station_anomalies = anomaly_dict[station_name]
    print(f"Found {len(station_anomalies)} anomaly time ranges for station '{station_name}'")
    
    # Create a mask for normal data points (not anomalies)
    print("Identifying anomaly data points...")
    normal_mask = np.ones(len(train_df), dtype=bool)
    
    # Convert timestamp column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(train_df['TimeStamp']):
        train_df['TimeStamp'] = pd.to_datetime(train_df['TimeStamp'])
    
    # Check each anomaly time range
    for anomaly_period in station_anomalies:
        # Handle tuple structure (assuming first element is start time, second is end time)
        if isinstance(anomaly_period, tuple) and len(anomaly_period) >= 2:
            start_time = pd.to_datetime(anomaly_period[0])
            end_time = pd.to_datetime(anomaly_period[1])
        else:
            print(f"Unexpected anomaly format: {anomaly_period}, skipping...")
            continue
        
        # Find rows that fall within this anomaly period
        anomaly_mask = (train_df['TimeStamp'] >= start_time) & (train_df['TimeStamp'] <= end_time)
        # Update the normal mask
        normal_mask = normal_mask & ~anomaly_mask
    
    # Filter out anomaly data points
    filtered_df = train_df[normal_mask]
    
    print(f"Removed {len(train_df) - len(filtered_df)} anomaly data points.")
    print(f"Filtered dataset has {len(filtered_df)} normal data points.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the filtered dataset
    output_path = os.path.join(output_dir, "train_normal.parquet")
    print(f"Saving filtered dataset to: {output_path}")
    filtered_df.to_parquet(output_path, index=False)
    
    print("Filtering complete!")
    
    return filtered_df

if __name__ == "__main__":
    train_path = "dataset/split_data/train.parquet"
    anomaly_dict_path = "dataset/origin_data/anomaly_dict_merged.pkl"
    
    filter_anomalies(train_path, anomaly_dict_path) 