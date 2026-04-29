import argparse
import csv
import glob
import os
import subprocess
import sys


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


def read_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    parsed = {"summary_path": os.path.abspath(summary_path)}
    for key, value in row.items():
        if key == "setting":
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            parsed[key] = value
    return parsed


def aggregate(rows, metrics):
    out = {}
    for metric in metrics:
        vals = [float(row[metric]) for row in rows if row.get(metric) != ""]
        if not vals:
            continue
        out[f"{metric}_mean"] = sum(vals) / len(vals)
        if len(vals) > 1:
            mean = out[f"{metric}_mean"]
            out[f"{metric}_std"] = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        else:
            out[f"{metric}_std"] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description="Run multi-split future-state classification comparisons.")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579",
    )
    parser.add_argument("--label_col", type=str, default="running_state_five_class")
    parser.add_argument(
        "--drop_cols",
        type=str,
        default="id,time,JianSuDuan_ChaoSu,running_state_class,running_state_five_class",
    )
    parser.add_argument("--label_shift", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--window_step", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--split_seeds", type=str, default="14,22,30")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--no_use_gpu", action="store_true", default=False)
    parser.add_argument("--output_root", type=str, default="/tmp/sgto_multisplit")
    parser.add_argument("--models", type=str, default="PatchTST,SGTONet,SGTONetV2,SGTONetV3,SGTONetV3Override,SGTONetV3Calibrated")
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
        "--data", "CSV_CLS",
        "--root_path", args.root_path,
        "--label_col", args.label_col,
        "--drop_cols", args.drop_cols,
        "--seq_len", str(args.seq_len),
        "--window_step", str(args.window_step),
        "--window_label_mode", "last",
        "--enable_future_state_targets",
        "--label_shift", str(args.label_shift),
        "--state_graph_profile", "hoister_overspeed",
        "--features", "M",
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--dropout", str(args.dropout),
        "--minority_raw_label", "9",
        "--use_class_weights",
        "--cls_loss", "ce",
        "--itr", "1",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--gpu", str(args.gpu),
        "--gpu_type", args.gpu_type,
        "--checkpoints", checkpoints_root,
        "--results_root", results_root,
    ]
    if args.no_use_gpu:
        common.append("--no_use_gpu")

    model_variants = {
        "PatchTST": {"args": ["--e_layers", "2", "--n_heads", "4", "--patch_len", "16"]},
        "iTransformer": {"args": ["--e_layers", "2", "--n_heads", "4"]},
        "TimesNet": {"args": ["--top_k", "3", "--num_kernels", "4"]},
        "DLinear": {"args": []},
        "SGTONet": {
            "args": [
                "--sgto_current_weight", "0.3",
                "--sgto_boundary_weight", "0.2",
                "--sgto_graph_weight", "0.02",
                "--sgto_align_weight", "0.1",
                "--sgto_boundary_soft_weight", "0.4",
                "--sgto_boundary_beta", "0.5",
            ]
        },
        "SGTONetV2": {
            "args": [
                "--sgto_current_weight", "0.3",
                "--sgto_boundary_weight", "0.2",
                "--sgto_graph_weight", "0.02",
                "--sgto_align_weight", "0.1",
                "--sgto_boundary_soft_weight", "0.4",
                "--sgto_boundary_beta", "0.5",
                "--sgto_proto_weight", "0.2",
                "--sgto_proto_sep_weight", "0.08",
                "--sgto_rare_gate_weight", "0.15",
                "--sgto_rare_pull_weight", "0.2",
                "--sgto_rare_pos_weight", "6.0",
                "--sgto_proto_margin", "0.6",
                "--sgto_proto_logit_scale", "10.0",
                "--sgto_proto_mix_weight", "0.45",
                "--sgto_rare_boost_scale", "1.6",
            ]
        },
        "SGTONetV3": {
            "args": [
                "--use_balanced_sampler",
                "--sampler_power", "1.0",
                "--minority_boost", "4.0",
                "--cls_loss", "focal",
                "--focal_gamma", "2.0",
                "--sgto_current_weight", "0.3",
                "--sgto_boundary_weight", "0.2",
                "--sgto_graph_weight", "0.02",
                "--sgto_align_weight", "0.1",
                "--sgto_boundary_soft_weight", "0.4",
                "--sgto_boundary_beta", "0.5",
                "--sgto_proto_weight", "0.2",
                "--sgto_proto_sep_weight", "0.08",
                "--sgto_rare_gate_weight", "0.12",
                "--sgto_rare_pull_weight", "0.2",
                "--sgto_rare_pos_weight", "8.0",
                "--sgto_rare_margin", "0.4",
                "--sgto_rare_margin_weight", "0.08",
                "--sgto_rare_align_weight", "0.15",
                "--sgto_proto_logit_scale", "10.0",
                "--sgto_proto_mix_weight", "0.35",
                "--sgto_rare_fuse_weight", "1.0",
                "--sgto_nonrare_suppress_weight", "0.1",
                "--sgto_rare_broad_gate",
                "--sgto_rare_precursor_labels", "5,7",
            ]
        },
        "SGTONetV3Override": {
            "model": "SGTONetV3",
            "args": [
                "--use_balanced_sampler",
                "--sampler_power", "1.0",
                "--minority_boost", "4.0",
                "--cls_loss", "focal",
                "--focal_gamma", "2.0",
                "--sgto_current_weight", "0.3",
                "--sgto_boundary_weight", "0.2",
                "--sgto_graph_weight", "0.02",
                "--sgto_align_weight", "0.1",
                "--sgto_boundary_soft_weight", "0.4",
                "--sgto_boundary_beta", "0.5",
                "--sgto_proto_weight", "0.2",
                "--sgto_proto_sep_weight", "0.08",
                "--sgto_rare_gate_weight", "0.12",
                "--sgto_rare_pull_weight", "0.2",
                "--sgto_rare_pos_weight", "8.0",
                "--sgto_rare_margin", "0.4",
                "--sgto_rare_margin_weight", "0.08",
                "--sgto_rare_align_weight", "0.15",
                "--sgto_proto_logit_scale", "10.0",
                "--sgto_proto_mix_weight", "0.35",
                "--sgto_rare_fuse_weight", "1.0",
                "--sgto_nonrare_suppress_weight", "0.1",
                "--sgto_rare_broad_gate",
                "--sgto_rare_precursor_labels", "5,7",
                "--sgto_rare_override",
                "--sgto_rare_override_threshold", "0.35",
                "--sgto_rare_override_precursor_labels", "5,7",
            ]
        },
        "SGTONetV3Calibrated": {
            "model": "SGTONetV3",
            "args": [
                "--use_balanced_sampler",
                "--sampler_power", "1.0",
                "--minority_boost", "4.0",
                "--cls_loss", "focal",
                "--focal_gamma", "2.0",
                "--sgto_current_weight", "0.3",
                "--sgto_boundary_weight", "0.2",
                "--sgto_graph_weight", "0.02",
                "--sgto_align_weight", "0.1",
                "--sgto_boundary_soft_weight", "0.4",
                "--sgto_boundary_beta", "0.5",
                "--sgto_proto_weight", "0.2",
                "--sgto_proto_sep_weight", "0.08",
                "--sgto_rare_gate_weight", "0.12",
                "--sgto_rare_pull_weight", "0.2",
                "--sgto_rare_pos_weight", "8.0",
                "--sgto_rare_margin", "0.4",
                "--sgto_rare_margin_weight", "0.08",
                "--sgto_rare_align_weight", "0.15",
                "--sgto_proto_logit_scale", "10.0",
                "--sgto_proto_mix_weight", "0.35",
                "--sgto_rare_fuse_weight", "1.0",
                "--sgto_nonrare_suppress_weight", "0.1",
                "--sgto_rare_broad_gate",
                "--sgto_rare_precursor_labels", "5,7",
                "--sgto_rare_override",
                "--sgto_rare_override_auto_threshold",
                "--sgto_rare_override_objective", "rare_f1",
                "--sgto_rare_override_precursor_labels", "5,7",
            ]
        },
    }

    ranked_args = list(model_variants["SGTONetV3"]["args"]) + [
        "--sgto_rare_rank_weight", "0.2",
        "--sgto_rare_rank_margin", "1.0",
        "--sgto_rare_hard_negative_labels", "5,7",
    ]
    model_variants["SGTONetV3Ranked"] = {
        "model": "SGTONetV3",
        "args": ranked_args,
    }
    model_variants["SGTONetV3RankedOverride"] = {
        "model": "SGTONetV3",
        "args": list(ranked_args) + [
            "--sgto_rare_override",
            "--sgto_rare_override_threshold", "0.35",
            "--sgto_rare_override_precursor_labels", "5,7",
        ],
    }
    model_variants["SGTONetV3RankedOverrideSoftmax"] = {
        "model": "SGTONetV3",
        "args": list(ranked_args) + [
            "--sgto_rare_override",
            "--sgto_rare_override_threshold", "0.35",
            "--sgto_rare_override_precursor_labels", "5,7",
            "--sgto_rare_override_min_softmax", "0.05",
        ],
    }
    v4_args = list(ranked_args) + [
        "--patch_len", "16",
        "--sgto_patch_stride", "8",
    ]
    model_variants["SGTONetV4"] = {
        "args": v4_args,
    }
    model_variants["SGTONetV4OverrideSoftmax"] = {
        "model": "SGTONetV4",
        "args": list(v4_args) + [
            "--sgto_rare_override",
            "--sgto_rare_override_threshold", "0.35",
            "--sgto_rare_override_precursor_labels", "5,7",
            "--sgto_rare_override_min_softmax", "0.05",
        ],
    }
    conservative_args = [
        "--cls_loss", "ce",
        "--sgto_current_weight", "0.3",
        "--sgto_boundary_weight", "0.2",
        "--sgto_graph_weight", "0.02",
        "--sgto_align_weight", "0.1",
        "--sgto_boundary_soft_weight", "0.4",
        "--sgto_boundary_beta", "0.5",
        "--sgto_proto_weight", "0.1",
        "--sgto_proto_sep_weight", "0.03",
        "--sgto_rare_gate_weight", "0.05",
        "--sgto_rare_pull_weight", "0.08",
        "--sgto_rare_pos_weight", "4.0",
        "--sgto_rare_margin", "0.4",
        "--sgto_rare_margin_weight", "0.04",
        "--sgto_rare_align_weight", "0.08",
        "--sgto_proto_logit_scale", "8.0",
        "--sgto_proto_mix_weight", "0.25",
        "--sgto_rare_fuse_weight", "0.5",
        "--sgto_nonrare_suppress_weight", "0.05",
        "--sgto_rare_rank_weight", "0.05",
        "--sgto_rare_rank_margin", "1.0",
        "--sgto_rare_hard_negative_labels", "5,7",
        "--patch_len", "16",
        "--sgto_patch_stride", "8",
        "--classification_early_stop_metric", "macro_f1",
    ]
    model_variants["SGTONetV4Conservative"] = {
        "model": "SGTONetV4",
        "args": conservative_args,
    }
    model_variants["SGTONetV4ConservativeOverride"] = {
        "model": "SGTONetV4",
        "args": list(conservative_args) + [
            "--sgto_rare_override",
            "--sgto_rare_override_threshold", "0.35",
            "--sgto_rare_override_precursor_labels", "5,7",
        ],
    }
    model_variants["SGTONetV5Conservative"] = {
        "model": "SGTONetV5",
        "args": conservative_args,
    }
    v6_args = list(conservative_args) + [
        "--sgto_rare_gate_weight", "0.2",
        "--sgto_rare_pos_weight", "8.0",
        "--sgto_rare_rank_weight", "0.2",
        "--sgto_dual_rare_fuse_weight", "0.0",
        "--sgto_dual_rare_suppress_weight", "0.0",
    ]
    model_variants["SGTONetV6Dual"] = {
        "model": "SGTONetV6",
        "args": v6_args,
    }
    model_variants["SGTONetV6DualOverride"] = {
        "model": "SGTONetV6",
        "args": list(v6_args) + [
            "--sgto_rare_override",
            "--sgto_rare_override_auto_threshold",
            "--sgto_rare_override_objective", "macro_f1",
            "--sgto_rare_override_threshold_min", "0.001",
            "--sgto_rare_override_threshold_max", "0.03",
            "--sgto_rare_override_threshold_steps", "30",
            "--sgto_rare_override_fallback_threshold", "0.01",
            "--sgto_rare_override_min_precision", "0.05",
            "--sgto_rare_override_precursor_labels", "5,7",
            "--sgto_rare_override_min_softmax", "0.0",
        ],
    }
    model_variants["SGTONetV6DualMeanContextOverride"] = {
        "model": "SGTONetV6",
        "args": list(v6_args) + [
            "--sgto_dual_rare_context", "mean",
            "--sgto_rare_override",
            "--sgto_rare_override_auto_threshold",
            "--sgto_rare_override_objective", "macro_f1",
            "--sgto_rare_override_threshold_min", "0.001",
            "--sgto_rare_override_threshold_max", "0.03",
            "--sgto_rare_override_threshold_steps", "30",
            "--sgto_rare_override_fallback_threshold", "0.01",
            "--sgto_rare_override_min_precision", "0.05",
            "--sgto_rare_override_precursor_labels", "5,7",
            "--sgto_rare_override_min_softmax", "0.0",
        ],
    }

    split_seeds = [int(item.strip()) for item in args.split_seeds.split(",") if item.strip()]
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "fault_macro_f1",
        "class9_precision",
        "class9_recall",
        "class9_f1",
        "rare_override_threshold",
        "rare_override_val_precision",
        "rare_override_val_recall",
        "rare_override_val_f1",
    ]

    raw_rows = []
    for split_seed in split_seeds:
        for model_name in selected_models:
            if model_name not in model_variants:
                raise ValueError(f"Unsupported model: {model_name}")
            variant = model_variants[model_name]
            actual_model_name = variant.get("model", model_name)
            des = f"futurecls_d{args.label_shift}_{model_name.lower()}_split{split_seed}"
            summary_path = None
            if args.skip_existing:
                try:
                    summary_path = find_summary(results_root, actual_model_name, des)
                except FileNotFoundError:
                    summary_path = None
            if summary_path is None:
                cmd = (
                    [sys.executable, "-u", "run.py"]
                    + common
                    + ["--model", actual_model_name, "--des", des, "--split_seed", str(split_seed)]
                    + variant["args"]
                )
                run_cmd(cmd, repo_root)
                summary_path = find_summary(results_root, actual_model_name, des)

            row = read_summary(summary_path)
            row["model"] = model_name
            row["split_seed"] = split_seed
            row["label_shift"] = args.label_shift
            raw_rows.append(row)

    raw_path = os.path.join(output_root, "future_state_multisplit_raw.csv")
    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        headers = ["model", "split_seed", "label_shift"] + metrics + ["summary_path", "setting"]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    agg_rows = []
    for model_name in selected_models:
        subset = [row for row in raw_rows if row["model"] == model_name]
        summary = {"model": model_name, "num_splits": len(subset), "label_shift": args.label_shift}
        summary.update(aggregate(subset, metrics))
        agg_rows.append(summary)
    agg_rows.sort(key=lambda row: row.get("macro_f1_mean", -1), reverse=True)

    agg_path = os.path.join(output_root, "future_state_multisplit_aggregate.csv")
    with open(agg_path, "w", encoding="utf-8", newline="") as f:
        headers = ["model", "num_splits", "label_shift"] + [f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "std")]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    print("\nSaved:")
    print(" -", raw_path)
    print(" -", agg_path)
    print("\nAggregate ranking:")
    for idx, row in enumerate(agg_rows, start=1):
        print(
            f"{idx}. {row['model']}: "
            f"macro_f1_mean={row.get('macro_f1_mean', 'NA')}, "
            f"balanced_accuracy_mean={row.get('balanced_accuracy_mean', 'NA')}, "
            f"class9_recall_mean={row.get('class9_recall_mean', 'NA')}"
        )


if __name__ == "__main__":
    main()
