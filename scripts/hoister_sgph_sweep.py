import argparse
import glob
import os
import subprocess
import sys


def run_cmd(cmd, cwd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def find_summary(results_root, des):
    pattern = os.path.join(
        results_root,
        f"classification_H5_SGPHNet_CSV_CLS_*_{des}_0",
        "summary.csv",
    )
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Cannot find summary.csv with pattern: {pattern}")
    return hits[-1]


def read_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    row = dict(zip(header, values))
    for key, value in list(row.items()):
        if key != "setting":
            try:
                row[key] = float(value)
            except ValueError:
                pass
    row["summary_path"] = os.path.abspath(summary_path)
    return row


def main():
    parser = argparse.ArgumentParser(description="Run a small SGPH-Net hyperparameter sweep on Hoister.")
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
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--split_seed", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.abspath(args.output_root)
    checkpoints_root = os.path.join(output_root, "checkpoints")
    results_root = os.path.join(output_root, "results")
    os.makedirs(checkpoints_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    common = [
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", "H5",
        "--model", "SGPHNet",
        "--data", "CSV_CLS",
        "--root_path", args.root_path,
        "--label_col", "running_state_five_class",
        "--drop_cols", "id,time,JianSuDuan_ChaoSu,running_state_class,running_state_five_class",
        "--seq_len", "96",
        "--window_step", "8",
        "--window_label_mode", "last",
        "--file_split_mode", "shuffle",
        "--split_seed", str(args.split_seed),
        "--train_ratio", "0.7",
        "--val_ratio", "0.15",
        "--features", "M",
        "--batch_size", str(args.batch_size),
        "--learning_rate", "0.001",
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--d_model", "128",
        "--d_ff", "256",
        "--dropout", "0.1",
        "--use_class_weights",
        "--cls_loss", "focal",
        "--focal_gamma", "2.0",
        "--enable_progression_targets",
        "--state_graph_profile", "hoister_overspeed",
        "--warning_horizon", "5",
        "--time_bucket_steps", "1,3,5,10",
        "--itr", "1",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--no_use_gpu",
        "--checkpoints", checkpoints_root,
        "--results_root", results_root,
    ]

    variants = [
        {
            "name": "sgph_light_aux",
            "args": [
                "--aux_hazard_weight", "0.2",
                "--aux_time_weight", "0.1",
                "--aux_next_state_weight", "0.1",
                "--aux_invalid_transition_weight", "0.02",
            ],
        },
        {
            "name": "sgph_sampler_b2",
            "args": [
                "--use_balanced_sampler",
                "--sampler_power", "1.0",
                "--minority_raw_label", "9",
                "--minority_boost", "2.0",
                "--aux_hazard_weight", "0.2",
                "--aux_time_weight", "0.1",
                "--aux_next_state_weight", "0.1",
                "--aux_invalid_transition_weight", "0.02",
            ],
        },
        {
            "name": "sgph_sampler_b3",
            "args": [
                "--use_balanced_sampler",
                "--sampler_power", "1.0",
                "--minority_raw_label", "9",
                "--minority_boost", "3.0",
                "--aux_hazard_weight", "0.2",
                "--aux_time_weight", "0.1",
                "--aux_next_state_weight", "0.1",
                "--aux_invalid_transition_weight", "0.02",
            ],
        },
    ]

    rows = []
    for variant in variants:
        des = variant["name"]
        summary_path = None
        if args.skip_existing:
            try:
                summary_path = find_summary(results_root, des)
            except FileNotFoundError:
                summary_path = None

        if summary_path is None:
            cmd = [sys.executable, "-u", "run.py"] + common + variant["args"] + ["--des", des]
            run_cmd(cmd, repo_root)
            summary_path = find_summary(results_root, des)

        row = read_summary(summary_path)
        row["variant"] = des
        rows.append(row)

    out_path = os.path.join(output_root, "sgph_sweep_summary.csv")
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(key, "")) for key in headers) + "\n")

    print("\nSaved:")
    print(" -", out_path)


if __name__ == "__main__":
    main()
