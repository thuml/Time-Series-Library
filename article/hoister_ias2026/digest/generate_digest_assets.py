from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[3]
DIGEST_DIR = Path(__file__).resolve().parent
FIG_DIR = DIGEST_DIR / "figures"
SOURCE_FIG_DIR = ROOT / "results" / "sgto_v6_dual" / "figures"


def add_box(ax, xy, width, height, text, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#202020",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
    )


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color="#303030",
        )
    )


def method_overview():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    blue = "#dcecff"
    green = "#dff3e3"
    yellow = "#fff0c2"
    red = "#ffd9d5"
    gray = "#eeeeee"

    add_box(ax, (0.03, 0.38), 0.17, 0.24, "Current sensor\nwindow", blue)
    add_box(ax, (0.27, 0.38), 0.18, 0.24, "Patch temporal\nencoder", green)
    add_box(ax, (0.54, 0.62), 0.18, 0.22, "Conservative\nfuture-state\nclassifier", yellow)
    add_box(ax, (0.54, 0.16), 0.18, 0.22, "Patch-attentive\nrare context\nand trigger", red)
    add_box(ax, (0.79, 0.38), 0.18, 0.24, "Boundary +\nprecursor gated\noverride", gray)

    add_arrow(ax, (0.20, 0.50), (0.27, 0.50))
    add_arrow(ax, (0.45, 0.50), (0.54, 0.72))
    add_arrow(ax, (0.45, 0.50), (0.54, 0.27))
    add_arrow(ax, (0.72, 0.73), (0.79, 0.55))
    add_arrow(ax, (0.72, 0.27), (0.79, 0.45))

    ax.text(
        0.50,
        0.04,
        r"SGTONet override fires only if rare score $\geq \tau$, boundary flag = 1, and current label $\in \{5,7\}$.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig0_method_overview.pdf"
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def copy_existing_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "fig1_main_d1_metrics.pdf",
        "fig2_class9_prf1.pdf",
        "fig3_ablation.pdf",
        "fig4_horizon_transfer.pdf",
        "fig5_confusion_v6_vs_itransformer.pdf",
        "fig6_threshold_sensitivity.pdf",
        "table_main_d1.tex",
    ]
    for name in names:
        src = SOURCE_FIG_DIR / name
        if src.exists():
            dst = FIG_DIR / name
            shutil.copy2(src, dst)
            print(f"Copied {dst}")
        else:
            print(f"Missing source figure: {src}")


def main():
    copy_existing_figures()
    method_overview()


if __name__ == "__main__":
    main()
