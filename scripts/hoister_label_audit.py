import argparse
import glob
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = "./dataset/Hoister/7-segment_id_only_jiansuduanchoasu_classification_5_13579"

SENSOR_GROUPS = {
    "speed": ["SuDuMoNiLiang", "FPLCSuDu", "BianmaQiSuDu", "CSJSuDu"],
    "depth": ["LGuanLongShenDu", "FPLCShenDu", "BianmaQiShenDu"],
    "current": ["DianshuDianliu1", "DianshuDianliu2", "LiCi_Current"],
    "load_pressure": ["ZhiDongPressure", "WZhuJiLiang", "WFuJiLiang"],
}


def read_csv_auto(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def list_data_files(root):
    files = sorted(glob.glob(os.path.join(root, "*.csv")))
    return [p for p in files if os.path.basename(p)[:1].isdigit()]


def contiguous_segments(df, label_col, file_name):
    labels = df[label_col].to_numpy()
    if len(labels) == 0:
        return []
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append(
                {
                    "file": file_name,
                    "label": labels[start],
                    "start_idx": start,
                    "end_idx": i - 1,
                    "length": i - start,
                }
            )
            start = i
    segments.append(
        {
            "file": file_name,
            "label": labels[start],
            "start_idx": start,
            "end_idx": len(labels) - 1,
            "length": len(labels) - start,
        }
    )
    return segments


def add_physics_features(df):
    out = df.copy()
    speed_cols = [c for c in SENSOR_GROUPS["speed"] if c in out.columns]
    depth_cols = [c for c in SENSOR_GROUPS["depth"] if c in out.columns]
    current_cols = [c for c in SENSOR_GROUPS["current"] if c in out.columns]

    if len(speed_cols) >= 2:
        out["feat_speed_range"] = out[speed_cols].max(axis=1) - out[speed_cols].min(axis=1)
        out["feat_speed_abs_mean"] = out[speed_cols].abs().mean(axis=1)
    if len(depth_cols) >= 2:
        out["feat_depth_range"] = out[depth_cols].max(axis=1) - out[depth_cols].min(axis=1)
    if {"DianshuDianliu1", "DianshuDianliu2"}.issubset(out.columns):
        out["feat_current_mismatch"] = (out["DianshuDianliu1"] - out["DianshuDianliu2"]).abs()
    if {"FPLCSuDu", "FPLCShenDu"}.issubset(out.columns):
        depth_delta = out["FPLCShenDu"].diff().abs()
        out["feat_speed_depth_residual"] = (depth_delta - out["FPLCSuDu"].abs()).abs()
    return out


def label_distribution(all_df, label_col):
    counts = all_df[label_col].value_counts().sort_index()
    total = counts.sum()
    rows = []
    for label, count in counts.items():
        rows.append({"label": label, "count": int(count), "percent": float(count / total)})
    return pd.DataFrame(rows)


def transition_table(segment_df):
    counter = Counter()
    file_segments = defaultdict(list)
    for row in segment_df.to_dict("records"):
        file_segments[row["file"]].append(row)
    for rows in file_segments.values():
        rows = sorted(rows, key=lambda r: r["start_idx"])
        for a, b in zip(rows[:-1], rows[1:]):
            counter[(a["label"], b["label"])] += 1
    return pd.DataFrame(
        [{"from": k[0], "to": k[1], "count": v} for k, v in counter.items()]
    ).sort_values(["count", "from", "to"], ascending=[False, True, True])


def segment_summary(segment_df, sample_seconds):
    rows = []
    for label, group in segment_df.groupby("label"):
        lengths = group["length"].to_numpy()
        rows.append(
            {
                "label": label,
                "segments": int(len(group)),
                "median_len_steps": float(np.median(lengths)),
                "mean_len_steps": float(np.mean(lengths)),
                "max_len_steps": int(np.max(lengths)),
                "median_seconds": float(np.median(lengths) * sample_seconds),
                "mean_seconds": float(np.mean(lengths) * sample_seconds),
            }
        )
    return pd.DataFrame(rows).sort_values("label")


def feature_summary(all_df, label_col):
    feature_cols = [c for c in all_df.columns if c.startswith("feat_")]
    rows = []
    for label, group in all_df.groupby(label_col):
        for col in feature_cols:
            vals = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "label": label,
                    "feature": col,
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std(ddof=0)),
                    "p90": float(vals.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "label"])


def fault_proximity(all_df, label_col, fault_col, sample_seconds):
    rows = []
    for file_name, group in all_df.groupby("__file"):
        group = group.reset_index(drop=True)
        fault_idx = np.flatnonzero(group[fault_col].to_numpy() == 1)
        if len(fault_idx) == 0:
            continue
        first_fault = int(fault_idx[0])
        before = group.iloc[: first_fault + 1].copy()
        before["steps_to_first_fault"] = first_fault - np.arange(len(before))
        for label, label_group in before.groupby(label_col):
            steps = label_group["steps_to_first_fault"].to_numpy()
            rows.append(
                {
                    "file": file_name,
                    "label": label,
                    "rows_before_or_at_fault": int(len(label_group)),
                    "min_seconds_to_fault": float(np.min(steps) * sample_seconds),
                    "median_seconds_to_fault": float(np.median(steps) * sample_seconds),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["label", "median_seconds_to_fault"])


def suspicious_short_segments(segment_df, min_len):
    return segment_df[segment_df["length"] <= min_len].sort_values(["label", "length", "file"])


def plot_label_timeline(all_df, out_path, label_col):
    files = sorted(all_df["__file"].unique())
    fig_h = max(5, len(files) * 0.28)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    label_values = sorted(all_df[label_col].unique())
    label_to_y = {label: i for i, label in enumerate(label_values)}
    cmap = plt.get_cmap("tab10")
    for row_idx, file_name in enumerate(files):
        group = all_df[all_df["__file"] == file_name].reset_index(drop=True)
        y = np.full(len(group), row_idx)
        colors = [cmap(label_to_y[v] % 10) for v in group[label_col].to_numpy()]
        ax.scatter(np.arange(len(group)), y, c=colors, s=3, marker="s")
    ax.set_yticks(np.arange(len(files)))
    ax.set_yticklabels(files, fontsize=7)
    ax.set_xlabel("row index")
    ax.set_title("Label timeline by file")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i % 10), label=str(label))
        for label, i in label_to_y.items()
    ]
    ax.legend(handles=handles, title=label_col, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_box(all_df, out_dir, label_col):
    feature_cols = [c for c in all_df.columns if c.startswith("feat_")]
    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = []
        labels = []
        for label, group in all_df.groupby(label_col):
            vals = group[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 0:
                data.append(vals.to_numpy())
                labels.append(str(label))
        if not data:
            plt.close(fig)
            continue
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("label")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{col}_boxplot.png"), dpi=180)
        plt.close(fig)


def df_to_markdown(df, floatfmt=".4f"):
    if df.empty:
        return "_Empty._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: format(x, floatfmt) if pd.notna(x) else "")
    headers = [str(c) for c in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(out_path, root, dist_df, seg_sum_df, trans_df, prox_df, feature_df, short_df):
    lines = []
    lines.append("# Hoister Label Audit Report")
    lines.append("")
    lines.append(f"Dataset: `{os.path.abspath(root)}`")
    lines.append("")
    lines.append("## What This Audit Checks")
    lines.append("")
    lines.append("- whether the five labels have stable temporal segments;")
    lines.append("- whether state transitions support a directed lifecycle story;")
    lines.append("- whether rare labels, especially class `9`, occur near the fault flag;")
    lines.append("- whether physics-inspired residual features separate labels.")
    lines.append("")
    lines.append("## Label Distribution")
    lines.append("")
    lines.append(df_to_markdown(dist_df, floatfmt=".4f"))
    lines.append("")
    lines.append("## Segment Duration")
    lines.append("")
    lines.append(df_to_markdown(seg_sum_df, floatfmt=".2f"))
    lines.append("")
    lines.append("## Most Frequent Transitions")
    lines.append("")
    lines.append(df_to_markdown(trans_df.head(20)))
    lines.append("")
    if not prox_df.empty:
        prox_summary = (
            prox_df.groupby("label", as_index=False)
            .agg(
                files_with_fault_context=("file", "nunique"),
                median_min_seconds_to_fault=("min_seconds_to_fault", "median"),
                median_seconds_to_fault=("median_seconds_to_fault", "median"),
            )
            .sort_values("label")
        )
        lines.append("## Proximity To First Fault Flag")
        lines.append("")
        lines.append(df_to_markdown(prox_summary, floatfmt=".2f"))
        lines.append("")
    lines.append("## Physics Residual Feature Summary")
    lines.append("")
    if feature_df.empty:
        lines.append("No residual feature summary was generated.")
    else:
        for feature, group in feature_df.groupby("feature"):
            lines.append(f"### {feature}")
            lines.append("")
            lines.append(df_to_markdown(group.drop(columns=["feature"]), floatfmt=".4f"))
            lines.append("")
    lines.append("## Short Segments To Inspect")
    lines.append("")
    if short_df.empty:
        lines.append("No short segments found under the configured threshold.")
    else:
        lines.append(df_to_markdown(short_df.head(80)))
    lines.append("")
    lines.append("## Immediate Interpretation Guide")
    lines.append("")
    lines.append("Keep the five-class protocol only if class `9` is consistently near fault-entry boundaries and has separable residual evidence from classes `5` and `7`.")
    lines.append("If class `9` is temporally random or indistinguishable from class `7`, downgrade it to a weak boundary label or merge it with class `7` for the main task, then report class `9` only as an exploratory rare-state analysis.")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Audit Hoister lifecycle labels before SGTO/SGTONet paper claims.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--label_col", type=str, default="running_state_five_class")
    parser.add_argument("--fault_col", type=str, default="JianSuDuan_ChaoSu")
    parser.add_argument("--sample_seconds", type=float, default=4.0)
    parser.add_argument("--short_segment_steps", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "label_audit")
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    segments = []
    for path in list_data_files(args.root):
        file_name = os.path.basename(path)
        df = read_csv_auto(path)
        if args.label_col not in df.columns:
            raise ValueError(f"{file_name} does not contain label column {args.label_col}")
        if args.fault_col not in df.columns:
            raise ValueError(f"{file_name} does not contain fault column {args.fault_col}")
        df = add_physics_features(df)
        df["__file"] = file_name
        frames.append(df)
        segments.extend(contiguous_segments(df, args.label_col, file_name))

    all_df = pd.concat(frames, ignore_index=True)
    segment_df = pd.DataFrame(segments)

    dist_df = label_distribution(all_df, args.label_col)
    seg_sum_df = segment_summary(segment_df, args.sample_seconds)
    trans_df = transition_table(segment_df)
    feature_df = feature_summary(all_df, args.label_col)
    prox_df = fault_proximity(all_df, args.label_col, args.fault_col, args.sample_seconds)
    short_df = suspicious_short_segments(segment_df, args.short_segment_steps)

    dist_df.to_csv(os.path.join(out_dir, "label_distribution.csv"), index=False)
    segment_df.to_csv(os.path.join(out_dir, "segments.csv"), index=False)
    seg_sum_df.to_csv(os.path.join(out_dir, "segment_summary.csv"), index=False)
    trans_df.to_csv(os.path.join(out_dir, "transitions.csv"), index=False)
    feature_df.to_csv(os.path.join(out_dir, "physics_feature_summary.csv"), index=False)
    prox_df.to_csv(os.path.join(out_dir, "fault_proximity.csv"), index=False)
    short_df.to_csv(os.path.join(out_dir, "short_segments_to_inspect.csv"), index=False)

    plot_label_timeline(all_df, os.path.join(out_dir, "label_timeline.png"), args.label_col)
    plot_feature_box(all_df, out_dir, args.label_col)

    report_path = os.path.join(out_dir, "LABEL_AUDIT_REPORT.md")
    write_markdown(report_path, args.root, dist_df, seg_sum_df, trans_df, prox_df, feature_df, short_df)

    print(f"Saved label audit to: {os.path.abspath(out_dir)}")
    print(f"Report: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
