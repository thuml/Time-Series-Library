import argparse
import glob
import os
import subprocess
import sys

import pandas as pd


def parse_seeds(seed_text):
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


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


def load_summary_row(summary_path):
    row = pd.read_csv(summary_path).iloc[0].to_dict()
    row["summary_path"] = os.path.abspath(summary_path)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed fault classification stability experiments.")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579",
    )
    parser.add_argument("--split_seed", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="2021,2022,2023,2024,2025")
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--strategies",
        type=str,
        default="baseline,sampler_focal",
        help="comma-separated strategy names in [baseline, sampler_ce, sampler_focal]",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="skip training when summary.csv already exists for a run",
    )
    parser.add_argument("--results_root", type=str, default="./results")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/fault_cls_stability",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    common = [
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", "H5",
        "--data", "CSV_CLS",
        "--root_path", args.root_path,
        "--label_col", "running_state_five_class",
        "--drop_cols", "id,time,JianSuDuan_ChaoSu,running_state_class,running_state_five_class",
        "--seq_len", "96",
        "--window_step", "8",
        "--file_split_mode", "shuffle",
        "--train_ratio", "0.7",
        "--val_ratio", "0.15",
        "--features", "M",
        "--learning_rate", "0.001",
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--itr", "1",
        "--num_workers", str(args.num_workers),
        "--split_seed", str(args.split_seed),
    ]

    models = [
        {
            "name": "SegRNN",
            "short": "sgrnn",
            "args": [
                "--model", "SegRNN",
                "--seg_len", "96",
                "--d_model", "64",
                "--dropout", "0.1",
                "--batch_size", "32",
            ],
        },
        {
            "name": "iTransformer",
            "short": "itr",
            "args": [
                "--model", "iTransformer",
                "--e_layers", "3",
                "--d_model", "64",
                "--d_ff", "128",
                "--dropout", "0.1",
                "--batch_size", "16",
            ],
        },
    ]

    strategy_args = {
        "baseline": ["--use_class_weights", "--cls_loss", "ce"],
        "sampler_ce": [
            "--use_balanced_sampler",
            "--sampler_power", "1.0",
            "--minority_raw_label", "9",
            "--minority_boost", "1.0",
            "--cls_loss", "ce",
        ],
        "sampler_focal": [
            "--use_balanced_sampler",
            "--sampler_power", "1.0",
            "--minority_raw_label", "9",
            "--minority_boost", "1.0",
            "--cls_loss", "focal",
            "--focal_gamma", "2.0",
        ],
    }
    strategy_code = {
        "baseline": "b",
        "sampler_ce": "sc",
        "sampler_focal": "sf",
    }

    for strategy in strategies:
        if strategy not in strategy_args:
            raise ValueError(f"Unsupported strategy: {strategy}")

    raw_rows = []
    for model in models:
        for strategy in strategies:
            for seed in seeds:
                des = f"stb_{model['short']}_{strategy_code[strategy]}_{seed}"
                summary_path = None
                if args.skip_existing:
                    try:
                        summary_path = find_summary(args.results_root, model["name"], des)
                    except FileNotFoundError:
                        summary_path = None

                if summary_path is None:
                    cmd = [sys.executable, "-u", "run.py"] + common + model["args"] + strategy_args[strategy] + [
                        "--seed", str(seed),
                        "--des", des,
                    ]
                    run_cmd(cmd, repo_root)
                    summary_path = find_summary(args.results_root, model["name"], des)

                row = load_summary_row(summary_path)
                row["model"] = model["name"]
                row["strategy"] = strategy
                row["seed"] = seed
                row["des"] = des
                raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_df = raw_df.sort_values(["model", "strategy", "seed"]).reset_index(drop=True)
    raw_path = os.path.join(args.out_dir, "raw_runs.csv")
    raw_df.to_csv(raw_path, index=False)

    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "fault_macro_f1",
        "class9_precision",
        "class9_recall",
        "class9_f1",
    ]
    agg_df = raw_df.groupby(["model", "strategy"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg_df.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        for col in agg_df.columns.to_flat_index()
    ]
    agg_path = os.path.join(args.out_dir, "aggregate_mean_std.csv")
    agg_df.to_csv(agg_path, index=False)

    print("\nSaved:")
    print(" -", os.path.abspath(raw_path))
    print(" -", os.path.abspath(agg_path))
    print("\nAggregate:")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
