import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

def load_minute_sampled_data(file_path: str) -> pd.DataFrame:
    """
    Load the minute-sampled parquet data
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Display basic info about the dataset
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def identify_time_column(df: pd.DataFrame) -> str:
    """
    Identify the time column in the dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        str: Name of the time column
    """
    logger.info("Identifying time column...")
    
    # Common time column names
    time_cols = ['timestamp', 'time', 'datetime', 'date', 'ts', 'Time', 'Timestamp', 'DateTime']
    
    # Check for explicit time column names
    for col in time_cols:
        if col in df.columns:
            logger.info(f"Found time column: {col}")
            return col
    
    # Check for columns with datetime-like data types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.info(f"Found datetime column: {col}")
            return col
    
    # Check for columns that can be converted to datetime
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(100))
            logger.info(f"Found convertible datetime column: {col}")
            return col
        except:
            continue
    
    # If index is datetime
    if pd.api.types.is_datetime64_any_dtype(df.index):
        logger.info("Using datetime index as time column")
        return 'index'
    
    logger.warning("No time column found. Please specify manually.")
    return None

def prepare_time_data(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Prepare and standardize time data
    
    Args:
        df: Input dataframe
        time_col: Name of the time column
        
    Returns:
        pd.DataFrame: Dataframe with prepared time data
    """
    logger.info(f"Preparing time data using column: {time_col}")
    
    df_copy = df.copy()
    
    if time_col == 'index':
        df_copy['timestamp'] = df_copy.index
        time_col = 'timestamp'
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
        logger.info("Converting to datetime...")
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    
    # Sort by time
    df_copy = df_copy.sort_values(time_col).reset_index(drop=True)
    
    logger.info(f"Time range: {df_copy[time_col].min()} to {df_copy[time_col].max()}")
    logger.info(f"Total time span: {df_copy[time_col].max() - df_copy[time_col].min()}")
    
    return df_copy, time_col

def analyze_time_continuity(df: pd.DataFrame, time_col: str) -> Dict:
    """
    Analyze time continuity of the dataset
    
    Args:
        df: Input dataframe with time data
        time_col: Name of the time column
        
    Returns:
        Dict: Analysis results
    """
    logger.info("Starting time continuity analysis...")
    
    # Basic statistics
    time_series = df[time_col]
    total_records = len(df)
    
    # Calculate time differences
    time_diffs = time_series.diff().dropna()
    
    # Expected interval (1 minute for minute-sampled data)
    expected_interval = timedelta(minutes=1)
    
    # Identify gaps
    gaps = time_diffs[time_diffs > expected_interval]
    
    # Analysis results
    results = {
        'total_records': total_records,
        'time_range': {
            'start': time_series.min(),
            'end': time_series.max(),
            'duration': time_series.max() - time_series.min()
        },
        'expected_interval': expected_interval,
        'actual_intervals': {
            'mean': time_diffs.mean(),
            'median': time_diffs.median(),
            'std': time_diffs.std(),
            'min': time_diffs.min(),
            'max': time_diffs.max()
        },
        'gaps': {
            'total_gaps': len(gaps),
            'gap_positions': gaps.index.tolist(),
            'gap_durations': gaps.values,
            'largest_gap': gaps.max() if len(gaps) > 0 else timedelta(0),
            'total_missing_time': gaps.sum() if len(gaps) > 0 else timedelta(0)
        },
        'continuity_stats': {}
    }
    
    # Calculate expected vs actual records
    if results['time_range']['duration'] > timedelta(0):
        expected_records = int(results['time_range']['duration'].total_seconds() / 60) + 1
        continuity_percentage = (total_records / expected_records) * 100
        missing_records = expected_records - total_records
    else:
        expected_records = total_records
        continuity_percentage = 100.0
        missing_records = 0
    
    results['continuity_stats'] = {
        'expected_records': expected_records,
        'actual_records': total_records,
        'missing_records': missing_records,
        'continuity_percentage': continuity_percentage
    }
    
    # Duplicate timestamps
    duplicates = time_series.duplicated().sum()
    results['duplicates'] = {
        'count': duplicates,
        'percentage': (duplicates / total_records) * 100
    }
    
    logger.info(f"Analysis completed. Found {len(gaps)} gaps in time series")
    logger.info(f"Continuity percentage: {continuity_percentage:.2f}%")
    
    return results

def generate_time_analysis_report(results: Dict) -> str:
    """
    Generate a comprehensive report of time analysis
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        str: Formatted report
    """
    logger.info("Generating time analysis report...")
    
    report = []
    report.append("=" * 80)
    report.append("TIME CONTINUITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Basic Information
    report.append("📊 BASIC INFORMATION")
    report.append("-" * 40)
    report.append(f"Total Records: {results['total_records']:,}")
    report.append(f"Time Range: {results['time_range']['start']} to {results['time_range']['end']}")
    report.append(f"Duration: {results['time_range']['duration']}")
    report.append(f"Expected Interval: {results['expected_interval']}")
    report.append("")
    
    # Continuity Statistics
    report.append("📈 CONTINUITY STATISTICS")
    report.append("-" * 40)
    report.append(f"Expected Records: {results['continuity_stats']['expected_records']:,}")
    report.append(f"Actual Records: {results['continuity_stats']['actual_records']:,}")
    report.append(f"Missing Records: {results['continuity_stats']['missing_records']:,}")
    report.append(f"Continuity Percentage: {results['continuity_stats']['continuity_percentage']:.2f}%")
    report.append("")
    
    # Interval Analysis
    report.append("⏱️ INTERVAL ANALYSIS")
    report.append("-" * 40)
    report.append(f"Mean Interval: {results['actual_intervals']['mean']}")
    report.append(f"Median Interval: {results['actual_intervals']['median']}")
    report.append(f"Std Deviation: {results['actual_intervals']['std']}")
    report.append(f"Min Interval: {results['actual_intervals']['min']}")
    report.append(f"Max Interval: {results['actual_intervals']['max']}")
    report.append("")
    
    # Gap Analysis
    report.append("🕳️ GAP ANALYSIS")
    report.append("-" * 40)
    report.append(f"Total Gaps: {results['gaps']['total_gaps']}")
    report.append(f"Largest Gap: {results['gaps']['largest_gap']}")
    report.append(f"Total Missing Time: {results['gaps']['total_missing_time']}")
    
    if results['gaps']['total_gaps'] > 0:
        report.append("\nTop 10 Largest Gaps:")
        gap_durations = sorted(results['gaps']['gap_durations'], reverse=True)[:10]
        for i, gap in enumerate(gap_durations, 1):
            report.append(f"  {i}. {gap}")
    report.append("")
    
    # Duplicate Analysis
    report.append("🔄 DUPLICATE ANALYSIS")
    report.append("-" * 40)
    report.append(f"Duplicate Timestamps: {results['duplicates']['count']}")
    report.append(f"Duplicate Percentage: {results['duplicates']['percentage']:.2f}%")
    report.append("")
    
    # Quality Assessment
    report.append("✅ QUALITY ASSESSMENT")
    report.append("-" * 40)
    continuity_pct = results['continuity_stats']['continuity_percentage']
    if continuity_pct >= 95:
        quality = "EXCELLENT"
        emoji = "🟢"
    elif continuity_pct >= 90:
        quality = "GOOD"
        emoji = "🟡"
    elif continuity_pct >= 80:
        quality = "FAIR"
        emoji = "🟠"
    else:
        quality = "POOR"
        emoji = "🔴"
    
    report.append(f"Overall Quality: {emoji} {quality}")
    report.append(f"Data Completeness: {continuity_pct:.2f}%")
    
    if results['gaps']['total_gaps'] == 0:
        report.append("✅ No gaps detected - Perfect continuity!")
    else:
        report.append(f"⚠️ {results['gaps']['total_gaps']} gaps detected")
    
    if results['duplicates']['count'] == 0:
        report.append("✅ No duplicate timestamps")
    else:
        report.append(f"⚠️ {results['duplicates']['count']} duplicate timestamps")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def save_analysis_results(results: Dict, report: str, output_dir: str = "dataset/analysis"):
    """
    Save analysis results to files
    
    Args:
        results: Analysis results dictionary
        report: Generated report string
        output_dir: Output directory for saving results
    """
    logger.info(f"Saving analysis results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as pickle
    import pickle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/time_continuity_results_{timestamp}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save report as text file
    with open(f"{output_dir}/time_continuity_report_{timestamp}.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Analysis results saved successfully")


def main():
    """
    Main function to perform time continuity analysis
    """
    # File path
    file_path = "dataset/data_preprocess/minute_sampled_Contacting_cleaned.parquet"
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    try:
        # Step 1: Load data
        df = load_minute_sampled_data(file_path)
        
        # Step 2: Identify time column
        time_col = identify_time_column(df)
        if time_col is None:
            logger.error("Could not identify time column. Please check your data.")
            return
        
        # Step 3: Prepare time data
        df_prepared, time_col = prepare_time_data(df, time_col)
        
        # Step 4: Analyze time continuity
        analysis_results = analyze_time_continuity(df_prepared, time_col)
        
        # Step 5: Generate report
        report = generate_time_analysis_report(analysis_results)
        
        # Step 6: Print report
        print(report)
        
        # Step 7: Save results
        save_analysis_results(analysis_results, report)

        
        logger.info("Time continuity analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Initialize logger
    logger = setup_logger("analyze_time_continuity")
    main() 