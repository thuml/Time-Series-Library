import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob


def plot_metrics(folder_path):
    """顯示 metrics.npy 內容"""
    metrics_path = os.path.join(folder_path, 'metrics.npy')
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True)
        # 根據 exp_long_term_forecasting.py 的儲存順序: [mae, mse, rmse, mape, mspe, r2, dtw]
        names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'R2', 'DTW']
        print("=" * 50)
        print("Metrics:")
        for i, val in enumerate(metrics):
            name = names[i] if i < len(names) else f"metric_{i}"
            print(f"  {name}: {val}")
        print("=" * 50)
    else:
        print(f"metrics.npy not found in {folder_path}")


def plot_predictions(folder_path, sample_indices=None, feature_idx=-1, max_samples=5):
    """
    畫出 pred.npy vs true.npy 的對比圖
    
    Args:
        folder_path: results 資料夾路徑
        sample_indices: 指定要畫的 sample index，None 則自動選取
        feature_idx: 要畫的 feature index，-1 代表最後一個 feature
        max_samples: 最多畫幾個 sample
    """
    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"pred.npy or true.npy not found in {folder_path}")
        return

    preds = np.load(pred_path)
    trues = np.load(true_path)
    print(f"Predictions shape: {preds.shape}")  # (num_samples, pred_len, features)
    print(f"Ground truth shape: {trues.shape}")

    # 選要畫的 sample
    num_samples = preds.shape[0]
    if sample_indices is None:
        step = max(1, num_samples // max_samples)
        sample_indices = list(range(0, num_samples, step))[:max_samples]

    n = len(sample_indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
    fig.suptitle(f'Prediction vs Ground Truth\n({os.path.basename(folder_path)})', fontsize=14)

    for row, idx in enumerate(sample_indices):
        ax = axes[row, 0]
        pred_vals = preds[idx, :, feature_idx]
        true_vals = trues[idx, :, feature_idx]

        ax.plot(true_vals, label='GroundTruth', linewidth=2, color='blue')
        ax.plot(pred_vals, label='Prediction', linewidth=2, color='red', linestyle='--')
        ax.set_title(f'Sample #{idx}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'prediction_plot.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(folder_path, 'prediction_plot.png')}")
    plt.show()


def plot_error_distribution(folder_path, feature_idx=-1):
    """畫出預測誤差的分布直方圖"""
    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"pred.npy or true.npy not found in {folder_path}")
        return

    preds = np.load(pred_path)
    trues = np.load(true_path)

    errors = (preds[:, :, feature_idx] - trues[:, :, feature_idx]).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Error Analysis\n({os.path.basename(folder_path)})', fontsize=14)

    # 誤差分布
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'Error Distribution (mean={errors.mean():.4f}, std={errors.std():.4f})')
    axes[0].set_xlabel('Error (Pred - True)')
    axes[0].set_ylabel('Count')

    # 每個 time step 的 MSE
    mse_per_step = np.mean((preds[:, :, feature_idx] - trues[:, :, feature_idx]) ** 2, axis=0)
    axes[1].bar(range(len(mse_per_step)), mse_per_step, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('MSE per Time Step')
    axes[1].set_xlabel('Prediction Step')
    axes[1].set_ylabel('MSE')

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(folder_path, 'error_analysis.png')}")
    plt.show()


def list_results():
    """列出所有 results 資料夾"""
    results_dirs = sorted(glob.glob('.\\results\\*\\'))
    if not results_dirs:
        print("No results found in .\\results\\")
        return []
    print("Available result folders:")
    for i, d in enumerate(results_dirs):
        has_pred = os.path.exists(os.path.join(d, 'pred.npy'))
        has_true = os.path.exists(os.path.join(d, 'true.npy'))
        has_metrics = os.path.exists(os.path.join(d, 'metrics.npy'))
        status = f"[pred:{'✓' if has_pred else '✗'} true:{'✓' if has_true else '✗'} metrics:{'✓' if has_metrics else '✗'}]"
        folder_name = os.path.basename(d.rstrip('\\'))
        print(f"  [{i}] {folder_name} {status}")
    return results_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot .npy prediction results')
    parser.add_argument('--folder', type=str, default=None,
                        help='Path to the result folder (e.g., ../results/xxx/)')
    parser.add_argument('--feature', type=int, default=-1,
                        help='Feature index to plot (-1 for last feature)')
    parser.add_argument('--samples', type=int, nargs='+', default=None,
                        help='Sample indices to plot (e.g., --samples 0 100 200)')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='Max number of samples to plot if --samples is not specified')
    parser.add_argument('--list', action='store_true',
                        help='List all available result folders')
    args = parser.parse_args()

    if args.list or args.folder is None:
        dirs = list_results()
        if args.list:
            exit(0)
        if not dirs:
            exit(1)
        # 互動式選擇
        try:
            choice = int(input("\nEnter folder index: "))
            args.folder = dirs[choice]
        except (ValueError, IndexError):
            print("Invalid choice.")
            exit(1)

    print(f"\nLoading results from: {args.folder}\n")
    plot_metrics(args.folder)
    plot_predictions(args.folder, sample_indices=args.samples,
                     feature_idx=args.feature, max_samples=args.max_samples)
    plot_error_distribution(args.folder, feature_idx=args.feature)