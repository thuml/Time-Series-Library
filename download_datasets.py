"""
下载 Time-Series-Library 数据集从 Hugging Face
"""
import os
from datasets import load_dataset
import pandas as pd

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
DATASET_ROOT = "./Time-Series-Library/dataset"

# 根据 Hugging Face 实际可用的配置
AVAILABLE_CONFIGS = [
    # ETT 数据集
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    # 主要预测数据集
    "electricity", "traffic", "weather", "exchange_rate", "national_illness",
    # M4 数据集
    "m4-yearly", "m4-quarterly", "m4-monthly", "m4-weekly", "m4-daily", "m4-hourly",
    # 分类数据集 (UEA)
    "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat",
    "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2",
    "SpokenArabicDigits", "UWaveGestureLibrary",
    # 异常检测数据集
    "SMD-data", "SMD-label", "MSL-data", "MSL-label",
    "SMAP-data", "SMAP-label", "PSM-data", "PSM-label", "SWaT"
]

def download_dataset(config_name, output_dir):
    """下载单个数据集并保存为CSV"""
    try:
        print(f"正在下载数据集: {config_name}")
        ds = load_dataset(HUGGINGFACE_REPO, name=config_name)

        # 保存训练集为CSV
        df = ds["train"].to_pandas()
        output_path = os.path.join(output_dir, f"{config_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"  ✓ 已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败 {config_name}: {e}")
        return False

def create_directory_structure():
    """创建数据集目录结构"""
    dirs = [
        os.path.join(DATASET_ROOT, "ETT-small"),
        os.path.join(DATASET_ROOT, "electricity"),
        os.path.join(DATASET_ROOT, "exchange_rate"),
        os.path.join(DATASET_ROOT, "traffic"),
        os.path.join(DATASET_ROOT, "weather"),
        os.path.join(DATASET_ROOT, "illness"),
        os.path.join(DATASET_ROOT, "m4"),
        os.path.join(DATASET_ROOT, "anomaly"),
        os.path.join(DATASET_ROOT, "classification"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"创建目录: {d}")

def main():
    print("=" * 60)
    print("Time-Series-Library 数据集下载工具")
    print("=" * 60)

    # 创建目录结构
    create_directory_structure()

    # 下载 ETT 数据集
    print("\n[1/5] 下载 ETT 数据集...")
    ett_configs = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    for config in ett_configs:
        download_dataset(config, os.path.join(DATASET_ROOT, "ETT-small"))

    # 下载主要预测数据集
    print("\n[2/5] 下载其他预测数据集...")
    forecast_configs = {
        "electricity": "electricity",
        "exchange_rate": "exchange_rate",
        "traffic": "traffic",
        "weather": "weather",
        "national_illness": "illness",
    }
    for config, folder in forecast_configs.items():
        download_dataset(config, os.path.join(DATASET_ROOT, folder))

    # 下载 M4 数据集
    print("\n[3/5] 下载 M4 数据集...")
    m4_configs = ["m4-yearly", "m4-quarterly", "m4-monthly", "m4-weekly", "m4-daily", "m4-hourly"]
    for config in m4_configs:
        download_dataset(config, os.path.join(DATASET_ROOT, "m4"))

    # 下载分类数据集
    print("\n[4/5] 下载分类数据集...")
    classification_configs = [
        "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2",
        "SpokenArabicDigits", "UWaveGestureLibrary"
    ]
    for config in classification_configs:
        download_dataset(config, os.path.join(DATASET_ROOT, "classification"))

    # 下载异常检测数据集
    print("\n[5/5] 下载异常检测数据集...")
    anomaly_configs = [
        "SMD-data", "SMD-label", "MSL-data", "MSL-label",
        "SMAP-data", "SMAP-label", "PSM-data", "PSM-label", "SWaT"
    ]
    for config in anomaly_configs:
        download_dataset(config, os.path.join(DATASET_ROOT, "anomaly"))

    print("\n" + "=" * 60)
    print("数据集下载完成!")
    print(f"数据集保存在: {os.path.abspath(DATASET_ROOT)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
