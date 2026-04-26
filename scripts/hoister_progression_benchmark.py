import argparse
import csv
import glob
import os
import subprocess
import sys


def parse_csv_items(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_seeds(seed_text):
    return [int(item.strip()) for item in str(seed_text).split(",") if item.strip()]


def run_cmd(cmd, cwd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_summary(results_root, model_name, des):
    pattern = os.path.join(
        results_root,
        f"classification_H5_{model_name}_CSV_CLS_*_{des}_0",
        "summary.csv",
    )
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Cannot find summary.csv with pattern: {pattern}")
    return hits[-1]


def read_summary_row(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Malformed summary file: {summary_path}")

    headers = lines[0].split(",")
    values = lines[1].split(",")
    row = {}
    for key, value in zip(headers, values):
        try:
            row[key] = float(value) if key != "setting" else value
        except ValueError:
            row[key] = value
    row["summary_path"] = os.path.abspath(summary_path)
    return row


def summarize_rows(rows, metric_keys):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["strategy"]), []).append(row)

    lines = []
    header = ["model", "strategy"]
    for metric in metric_keys:
        header.extend([f"{metric}_mean", f"{metric}_std"])
    lines.append(",".join(header))

    for (model, strategy), group_rows in sorted(grouped.items()):
        parts = [model, strategy]
        for metric in metric_keys:
            vals = [float(r[metric]) for r in group_rows if metric in r]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                std = var ** 0.5
            else:
                std = 0.0
            parts.extend([f"{mean:.6f}", f"{std:.6f}"])
        lines.append(",".join(parts))
    return "\n".join(lines) + "\n"


def load_existing_rows(raw_csv_path):
    if not os.path.exists(raw_csv_path):
        return []

    rows = []
    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"setting", "summary_path", "model", "strategy", "des"}:
                    parsed[key] = value
                elif key == "seed":
                    parsed[key] = int(float(value))
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows.append(parsed)
    return rows


def merge_rows(existing_rows, new_rows):
    merged = {}
    for row in existing_rows + new_rows:
        key = (row.get("model"), row.get("strategy"), row.get("seed"), row.get("des"))
        merged[key] = row
    return [merged[key] for key in sorted(merged)]


def main():
    parser = argparse.ArgumentParser(description="Run Hoister progression benchmarks with consistent settings.")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579/experiment_outputs",
    )
    parser.add_argument("--models", type=str, default="SegRNN,iTransformer,Transformer,SGPHNet")
    parser.add_argument("--seeds", type=str, default="2,3,4")
    parser.add_argument("--split_seed", type=int, default=2)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--window_step", type=int, default=8)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="skip runs that already have summary.csv",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.abspath(args.output_root)
    checkpoints_root = os.path.join(output_root, "checkpoints")
    results_root = os.path.join(output_root, "results")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    model_names = parse_csv_items(args.models)
    seeds = parse_seeds(args.seeds)

    common = [
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", "H5",
        "--data", "CSV_CLS",
        "--root_path", args.root_path,
        "--label_col", "running_state_five_class",
        "--drop_cols", "id,time,JianSuDuan_ChaoSu,running_state_class,running_state_five_class",
        "--seq_len", str(args.seq_len),
        "--window_step", str(args.window_step),
        "--window_label_mode", "last",
        "--file_split_mode", "shuffle",
        "--train_ratio", "0.7",
        "--val_ratio", "0.15",
        "--features", "M",
        "--learning_rate", str(args.learning_rate),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--itr", "1",
        "--num_workers", str(args.num_workers),
        "--split_seed", str(args.split_seed),
        "--checkpoints", checkpoints_root,
        "--results_root", results_root,
    ]

    model_configs = {
        "SegRNN": {
            "args": ["--model", "SegRNN", "--seg_len", str(args.seq_len), "--d_model", "64", "--dropout", "0.1",
                     "--batch_size", str(args.batch_size)],
            "strategy": "flat_focal",
            "extra": ["--use_class_weights", "--cls_loss", "focal", "--focal_gamma", "2.0"],
        },
        "iTransformer": {
            "args": ["--model", "iTransformer", "--e_layers", "3", "--d_model", "64", "--d_ff", "128",
                     "--dropout", "0.1", "--batch_size", str(max(16, args.batch_size // 2))],
            "strategy": "flat_focal",
            "extra": ["--use_class_weights", "--cls_loss", "focal", "--focal_gamma", "2.0"],
        },
        "Transformer": {
            "args": ["--model", "Transformer", "--e_layers", "2", "--d_model", "64", "--d_ff", "128",
                     "--dropout", "0.1", "--batch_size", str(max(16, args.batch_size // 2))],
            "strategy": "flat_focal",
            "extra": ["--use_class_weights", "--cls_loss", "focal", "--focal_gamma", "2.0"],
        },
        "SGPHNet": {
            "args": ["--model", "SGPHNet", "--d_model", "128", "--d_ff", "256", "--dropout", "0.1",
                     "--batch_size", str(args.batch_size)],
            "strategy": "graph_prog",
            "extra": [
                "--use_class_weights",
                "--cls_loss", "focal",
                "--focal_gamma", "2.0",
                "--enable_progression_targets",
                "--state_graph_profile", "hoister_overspeed",
                "--warning_horizon", "5",
                "--time_bucket_steps", "1,3,5,10",
                "--aux_hazard_weight", "0.5",
                "--aux_time_weight", "0.3",
                "--aux_next_state_weight", "0.3",
                "--aux_invalid_transition_weight", "0.05",
            ],
        },
    }

    raw_csv_path = os.path.join(output_root, "raw_runs.csv")
    existing_rows = load_existing_rows(raw_csv_path)
    raw_rows = []
    for model_name in model_names:
        if model_name not in model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        model_cfg = model_configs[model_name]

        for seed in seeds:
            des = f"hoister_{model_name.lower()}_{seed}"
            summary_path = None
            if args.skip_existing:
                try:
                    summary_path = find_summary(results_root, model_name, des)
                except FileNotFoundError:
                    summary_path = None

            if summary_path is None:
                cmd = [sys.executable, "-u", "run.py"] + common + model_cfg["args"] + model_cfg["extra"] + [
                    "--seed", str(seed),
                    "--des", des,
                    "--no_use_gpu",
                ]
                run_cmd(cmd, repo_root)
                summary_path = find_summary(results_root, model_name, des)

            row = read_summary_row(summary_path)
            row["model"] = model_name
            row["strategy"] = model_cfg["strategy"]
            row["seed"] = seed
            row["des"] = des
            raw_rows.append(row)

    metric_keys = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "fault_macro_f1",
        "class9_precision",
        "class9_recall",
        "class9_f1",
    ]

    final_rows = merge_rows(existing_rows, raw_rows)

    if final_rows:
        headers = list(final_rows[0].keys())
        with open(raw_csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in final_rows:
                vals = [str(row.get(key, "")) for key in headers]
                f.write(",".join(vals) + "\n")

    aggregate_csv_path = os.path.join(output_root, "aggregate_mean_std.csv")
    with open(aggregate_csv_path, "w", encoding="utf-8") as f:
        f.write(summarize_rows(final_rows, metric_keys))

    print("\nSaved:")
    print(" -", raw_csv_path)
    print(" -", aggregate_csv_path)


if __name__ == "__main__":
    main()
