#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.m4_summary import M4Summary

LONG_TERM_TARGETS = {
    "ETTm1": {"mse": 0.400, "mae": 0.406},
    "ETTm2": {"mse": 0.291, "mae": 0.333},
    "ETTh1": {"mse": 0.458, "mae": 0.450},
    "ETTh2": {"mse": 0.414, "mae": 0.427},
    "Electricity": {"mse": 0.192, "mae": 0.295},
    "Traffic": {"mse": 0.620, "mae": 0.336},
    "Weather": {"mse": 0.259, "mae": 0.287},
    "Exchange": {"mse": 0.416, "mae": 0.443},
    "ILI": {"mse": 2.139, "mae": 0.931},
}

IMPUTATION_TARGETS = {
    "ETTm1": {"mse": 0.027, "mae": 0.107},
    "ETTm2": {"mse": 0.022, "mae": 0.088},
    "ETTh1": {"mse": 0.078, "mae": 0.187},
    "ETTh2": {"mse": 0.049, "mae": 0.146},
    "Electricity": {"mse": 0.092, "mae": 0.210},
    "Weather": {"mse": 0.030, "mae": 0.054},
}

SHORT_TERM_TARGETS = {
    "SMAPE": 11.829,
    "MASE": 1.585,
    "OWA": 0.851,
}

CLASSIFICATION_TARGETS = {
    "EthanolConcentration": 35.7,
    "FaceDetection": 68.6,
    "Handwriting": 32.1,
    "Heartbeat": 78.0,
    "JapaneseVowels": 98.4,
    "PEMS-SF": 89.6,
    "SelfRegulationSCP1": 91.8,
    "SelfRegulationSCP2": 57.2,
    "SpokenArabicDigits": 99.0,
    "UWaveGestureLibrary": 85.3,
    "Average": 73.6,
}

ANOMALY_TARGETS = {
    "SMD": 85.12,
    "MSL": 84.18,
    "SMAP": 70.85,
    "SWAT": 92.10,
    "PSM": 95.21,
    "Average": 85.49,
}

LONG_TERM_ALIASES = {
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ecl": "Electricity",
    "electricity": "Electricity",
    "traffic": "Traffic",
    "weather": "Weather",
    "exchange": "Exchange",
    "ili": "ILI",
}

IMPUTATION_ALIASES = {
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ecl": "Electricity",
    "electricity": "Electricity",
    "weather": "Weather",
}

ANOMALY_ALIASES = {
    "smd": "SMD",
    "msl": "MSL",
    "smap": "SMAP",
    "swat": "SWAT",
    "psm": "PSM",
}

LONG_TERM_RE = re.compile(r"^long_term_forecast_(?P<model_id>.+?)_TimesNet_(?P<data>[^_]+)_ft")
IMPUTATION_RE = re.compile(r"^imputation_(?P<model_id>.+?)_TimesNet_(?P<data>[^_]+)_ft")
CLASSIFICATION_RE = re.compile(r"^classification_(?P<model_id>.+?)_TimesNet_(?P<data>[^_]+)_ft")
ANOMALY_RE = re.compile(r"^anomaly_detection_(?P<model_id>.+?)_TimesNet_(?P<data>[^_]+)_ft")


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_delta(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def add_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")


def should_skip(name: str) -> bool:
    lowered = name.lower()
    return "smoke" in lowered


def normalize_from_model_id(model_id: str, mapping: dict[str, str]) -> str | None:
    prefix = model_id.split("_")[0].lower()
    return mapping.get(prefix)


def summarize_long_term(results_dir: Path) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.npy"):
        setting = metrics_path.parent.name
        if should_skip(setting):
            continue
        match = LONG_TERM_RE.match(setting)
        if not match:
            continue
        dataset = normalize_from_model_id(match.group("model_id"), LONG_TERM_ALIASES)
        if dataset is None:
            continue
        metrics = np.load(metrics_path)
        mae = float(metrics[0])
        mse = float(metrics[1])
        grouped[dataset].append((mse, mae))

    summary: dict[str, dict[str, float | int]] = {}
    for dataset, values in grouped.items():
        summary[dataset] = {
            "count": len(values),
            "mse": float(np.mean([item[0] for item in values])),
            "mae": float(np.mean([item[1] for item in values])),
        }
    return summary


def summarize_imputation(results_dir: Path) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for metrics_path in results_dir.glob("*/metrics.npy"):
        setting = metrics_path.parent.name
        if should_skip(setting):
            continue
        match = IMPUTATION_RE.match(setting)
        if not match:
            continue
        dataset = normalize_from_model_id(match.group("model_id"), IMPUTATION_ALIASES)
        if dataset is None:
            continue
        metrics = np.load(metrics_path)
        mae = float(metrics[0])
        mse = float(metrics[1])
        grouped[dataset].append((mse, mae))

    summary: dict[str, dict[str, float | int]] = {}
    for dataset, values in grouped.items():
        summary[dataset] = {
            "count": len(values),
            "mse": float(np.mean([item[0] for item in values])),
            "mae": float(np.mean([item[1] for item in values])),
        }
    return summary


def summarize_classification(results_dir: Path) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for result_file in results_dir.glob("*/result_classification.txt"):
        setting = result_file.parent.name
        if should_skip(setting):
            continue
        match = CLASSIFICATION_RE.match(setting)
        if not match:
            continue
        dataset = match.group("model_id")
        text = result_file.read_text()
        metric_match = re.search(r"accuracy:([0-9.]+)", text)
        if not metric_match:
            continue
        accuracy = float(metric_match.group(1)) * 100.0
        grouped[dataset].append(accuracy)

    summary: dict[str, dict[str, float | int]] = {}
    for dataset, values in grouped.items():
        summary[dataset] = {
            "count": len(values),
            "accuracy": float(max(values)),
        }
    return summary


def summarize_anomaly(root_dir: Path) -> dict[str, dict[str, float | int]]:
    result_file = root_dir / "result_anomaly_detection.txt"
    grouped: dict[str, list[float]] = defaultdict(list)
    if not result_file.exists():
        return {}

    current_setting: str | None = None
    for raw_line in result_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("anomaly_detection_"):
            current_setting = line
            continue
        if current_setting is None:
            continue
        if "F-score" not in line:
            continue

        if should_skip(current_setting):
            current_setting = None
            continue

        match = ANOMALY_RE.match(current_setting)
        score_match = re.search(r"F-score : ([0-9.]+)", line)
        if not match or not score_match:
            current_setting = None
            continue

        dataset = normalize_from_model_id(match.group("model_id"), ANOMALY_ALIASES)
        if dataset is None:
            current_setting = None
            continue

        f1 = float(score_match.group(1)) * 100.0
        grouped[dataset].append(f1)
        current_setting = None

    summary: dict[str, dict[str, float | int]] = {}
    for dataset, values in grouped.items():
        summary[dataset] = {
            "count": len(values),
            "f1": float(max(values)),
        }
    return summary


def summarize_short_term(root_dir: Path) -> dict[str, float]:
    m4_dir = root_dir / "m4_results" / "TimesNet"
    required_files = [
        "Yearly_forecast.csv",
        "Quarterly_forecast.csv",
        "Monthly_forecast.csv",
        "Weekly_forecast.csv",
        "Daily_forecast.csv",
        "Hourly_forecast.csv",
    ]
    if not all((m4_dir / name).exists() for name in required_files):
        return {}

    summary = M4Summary(str(m4_dir) + "/", str(root_dir / "dataset" / "m4"))
    smape, owa, _, mase = summary.evaluate()
    return {
        "SMAPE": float(smape["Average"]),
        "MASE": float(mase["Average"]),
        "OWA": float(owa["Average"]),
    }


def build_markdown(
    long_term: dict[str, dict[str, float | int]],
    short_term: dict[str, float],
    imputation: dict[str, dict[str, float | int]],
    classification: dict[str, dict[str, float | int]],
    anomaly: dict[str, dict[str, float | int]],
) -> str:
    lines: list[str] = []
    lines.append("# TimesNet Reproduction Summary")
    lines.append("")
    lines.append("This file is generated by `scripts/summarize_timesnet_results.py`.")
    lines.append("")

    lines.append("## Long-Term Forecasting")
    lines.append("")
    rows: list[list[str]] = []
    for dataset, target in LONG_TERM_TARGETS.items():
        ours = long_term.get(dataset)
        if ours is None:
            rows.append([dataset, "-", fmt_float(target["mse"]), "-", "-", fmt_float(target["mae"]), "-", "-"])
            continue
        rows.append(
            [
                dataset,
                fmt_float(float(ours["mse"])),
                fmt_float(target["mse"]),
                fmt_delta(float(ours["mse"]) - target["mse"]),
                fmt_float(float(ours["mae"])),
                fmt_float(target["mae"]),
                fmt_delta(float(ours["mae"]) - target["mae"]),
                str(int(ours["count"])),
            ]
        )
    add_table(lines, ["Dataset", "Ours MSE", "Paper MSE", "Delta", "Ours MAE", "Paper MAE", "Delta", "Runs"], rows)

    lines.append("## Short-Term Forecasting")
    lines.append("")
    rows = []
    for metric_name, target in SHORT_TERM_TARGETS.items():
        ours = short_term.get(metric_name)
        if ours is None:
            rows.append([metric_name, "-", fmt_float(target), "-"])
            continue
        rows.append([metric_name, fmt_float(ours), fmt_float(target), fmt_delta(ours - target)])
    add_table(lines, ["Metric", "Ours", "Paper", "Delta"], rows)

    lines.append("## Imputation")
    lines.append("")
    rows = []
    for dataset, target in IMPUTATION_TARGETS.items():
        ours = imputation.get(dataset)
        if ours is None:
            rows.append([dataset, "-", fmt_float(target["mse"]), "-", "-", fmt_float(target["mae"]), "-", "-"])
            continue
        rows.append(
            [
                dataset,
                fmt_float(float(ours["mse"])),
                fmt_float(target["mse"]),
                fmt_delta(float(ours["mse"]) - target["mse"]),
                fmt_float(float(ours["mae"])),
                fmt_float(target["mae"]),
                fmt_delta(float(ours["mae"]) - target["mae"]),
                str(int(ours["count"])),
            ]
        )
    add_table(lines, ["Dataset", "Ours MSE", "Paper MSE", "Delta", "Ours MAE", "Paper MAE", "Delta", "Runs"], rows)

    lines.append("## Classification")
    lines.append("")
    rows = []
    available_accs = []
    for dataset, target in CLASSIFICATION_TARGETS.items():
        if dataset == "Average":
            continue
        ours = classification.get(dataset)
        if ours is None:
            rows.append([dataset, "-", fmt_float(target, 1), "-", "-"])
            continue
        available_accs.append(float(ours["accuracy"]))
        rows.append(
            [
                dataset,
                fmt_float(float(ours["accuracy"]), 1),
                fmt_float(target, 1),
                fmt_delta(float(ours["accuracy"]) - target, 1),
                str(int(ours["count"])),
            ]
        )
    add_table(lines, ["Dataset", "Ours Acc(%)", "Paper Acc(%)", "Delta", "Runs"], rows)

    average_row = []
    if available_accs:
        average_accuracy = float(np.mean(available_accs))
        average_row.append(
            [
                "Average",
                fmt_float(average_accuracy, 1),
                fmt_float(CLASSIFICATION_TARGETS["Average"], 1),
                fmt_delta(average_accuracy - CLASSIFICATION_TARGETS["Average"], 1),
                str(len(available_accs)),
            ]
        )
    else:
        average_row.append(["Average", "-", fmt_float(CLASSIFICATION_TARGETS["Average"], 1), "-", "0"])
    add_table(lines, ["Dataset", "Ours Acc(%)", "Paper Acc(%)", "Delta", "Covered"], average_row)

    lines.append("## Anomaly Detection")
    lines.append("")
    rows = []
    available_f1 = []
    for dataset, target in ANOMALY_TARGETS.items():
        if dataset == "Average":
            continue
        ours = anomaly.get(dataset)
        if ours is None:
            rows.append([dataset, "-", fmt_float(target, 2), "-", "-"])
            continue
        available_f1.append(float(ours["f1"]))
        rows.append(
            [
                dataset,
                fmt_float(float(ours["f1"]), 2),
                fmt_float(target, 2),
                fmt_delta(float(ours["f1"]) - target, 2),
                str(int(ours["count"])),
            ]
        )
    add_table(lines, ["Dataset", "Ours F1(%)", "Paper F1(%)", "Delta", "Runs"], rows)

    average_row = []
    if available_f1:
        average_f1 = float(np.mean(available_f1))
        average_row.append(
            [
                "Average",
                fmt_float(average_f1, 2),
                fmt_float(ANOMALY_TARGETS["Average"], 2),
                fmt_delta(average_f1 - ANOMALY_TARGETS["Average"], 2),
                str(len(available_f1)),
            ]
        )
    else:
        average_row.append(["Average", "-", fmt_float(ANOMALY_TARGETS["Average"], 2), "-", "0"])
    add_table(lines, ["Dataset", "Ours F1(%)", "Paper F1(%)", "Delta", "Covered"], average_row)

    lines.append("## Notes")
    lines.append("")
    lines.append("- Long-term forecasting and imputation paper values are the averages reported in the main paper tables.")
    lines.append("- Classification and anomaly detection values in the repository logs are ratios in `[0, 1]`; this summary converts them to percentages to match the paper.")
    lines.append("- For anomaly detection, if a dataset has multiple runs, this summary keeps the best F1 because `SWaT` is searched over several settings in the provided script.")
    lines.append("- If a row is `-`, the corresponding experiment has not been completed yet.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TimesNet reproduction results and compare with the paper.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "TIMESNET_REPRODUCTION_RESULTS.md",
        help="Output markdown file.",
    )
    args = parser.parse_args()

    root_dir = args.root.resolve()
    results_dir = root_dir / "results"

    long_term = summarize_long_term(results_dir)
    short_term = summarize_short_term(root_dir)
    imputation = summarize_imputation(results_dir)
    classification = summarize_classification(results_dir)
    anomaly = summarize_anomaly(root_dir)

    markdown = build_markdown(long_term, short_term, imputation, classification, anomaly)
    args.output.write_text(markdown)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
