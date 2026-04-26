import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_FILE = os.path.join(ROOT_DIR, "experiment_log.csv")
OUTPUT_DIR = SCRIPT_DIR

TARGET_RUNS = [
    {
        "label": "Full KD",
        "group": "Baseline",
        "model": "real_wikitext_full_kd",
        "budget": 50277,
        "epochs": 3,
        "lr": 5e-05,
    },
    {
        "label": "Top-4 KD",
        "group": "Top-K",
        "model": "real_wikitext_topk_k4",
        "budget": 8,
        "epochs": 3,
        "lr": 5e-05,
    },
    {
        "label": "Top-8 KD",
        "group": "Top-K",
        "model": "real_wikitext_topk_k8",
        "budget": 16,
        "epochs": 3,
        "lr": 5e-05,
    },
    {
        "label": "Top-16 KD",
        "group": "Top-K",
        "model": "real_topk_k16",
        "budget": 32,
        "epochs": 1,
        "lr": 5e-05,
    },
    {
        "label": "Sampling KD",
        "group": "Baseline",
        "model": "real_wikitext_sampling_k50",
        "budget": 100,
        "epochs": 3,
        "lr": 6e-05,
    },
    {
        "label": "Adapt. Top-K",
        "group": "Ours",
        "model": "real_wikitext_adaptive_topk",
        "budget": 20,
        "epochs": 3,
        "lr": 5e-05,
    },
    {
        "label": "+ Head-Mass Wt.",
        "group": "Ours",
        "model": "real_wikitext_adaptive_topk_weighted",
        "budget": 20,
        "epochs": 3,
        "lr": 5e-05,
    },
]

STYLE = {
    "paper": "#ffffff",
    "axes": "#ffffff",
    "ink": "#20242a",
    "muted": "#66717c",
    "grid": "#d7dce2",
    "topk": "#77b7e5",
    "sampling": "#2bb7a8",
    "full": "#e31a1c",
    "adapt": "#d915d8",
    "head": "#5717f2",
}

METHOD_STYLE = {
    "Top-4 KD": {"color": STYLE["topk"], "marker": "o", "size": 88},
    "Top-8 KD": {"color": STYLE["topk"], "marker": "o", "size": 88},
    "Top-16 KD": {"color": STYLE["topk"], "marker": "o", "size": 88},
    "Sampling KD": {"color": STYLE["sampling"], "marker": "D", "size": 92},
    "Full KD": {"color": STYLE["full"], "marker": "s", "size": 92},
    "Adapt. Top-K": {"color": STYLE["adapt"], "marker": "^", "size": 130},
    "+ Head-Mass Wt.": {"color": STYLE["head"], "marker": "p", "size": 150},
}

LEGEND_LABELS = {
    "Full KD": "Full",
    "Top-K": r"Top-$K$ ($K \in \{4, 8, 16\}$)",
    "Sampling KD": "Sampling",
    "Adapt. Top-K": "Adaptive Top-k (Ours)",
    "+ Head-Mass Wt.": "Adaptive Top-k with Head-Mass Weighting (Ours)",
}


def load_experiment_rows(log_file):
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["method"]]


def find_target_row(rows, target):
    for row in rows:
        if row["model"] != target["model"]:
            continue
        if int(row["budget"]) != target["budget"]:
            continue
        if int(row["epochs"]) != target["epochs"]:
            continue
        if float(row["lr"]) != target["lr"]:
            continue
        return row
    raise ValueError(f"Could not find CSV row for {target['label']}")


def build_plot_rows(log_file):
    csv_rows = load_experiment_rows(log_file)
    plot_rows = []
    for target in TARGET_RUNS:
        row = find_target_row(csv_rows, target)
        plot_rows.append(
            {
                "method": target["label"],
                "group": target["group"],
                "budget": int(row["budget"]),
                "nll": float(row["val_nll"]),
                "ppl": float(row["val_ppl"]),
            }
        )
    return plot_rows


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.solid_capstyle": "round",
            "mathtext.fontset": "cm",
        }
    )


def compact_budget(value):
    if value >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return str(value)


def style_axis(ax):
    ax.set_facecolor(STYLE["axes"])
    ax.grid(True, which="major", axis="both", color=STYLE["grid"], linewidth=0.85, alpha=0.72)
    ax.grid(True, which="minor", axis="x", color=STYLE["grid"], linewidth=0.45, alpha=0.28)
    ax.tick_params(colors=STYLE["muted"], length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_color("#c9ced6")
        spine.set_linewidth(0.9)


def y_limits(rows, metric):
    values = [row[metric] for row in rows]
    low = min(values)
    high = max(values)
    pad = (high - low) * 0.08
    return low - pad, high + pad


def plot_metric(rows, metric, label, output_dir, output_name):
    topk_rows = sorted([row for row in rows if row["group"] == "Top-K"], key=lambda row: row["budget"])
    best = min(rows, key=lambda row: row[metric])

    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=STYLE["paper"])
    style_axis(ax)

    ax.set_xscale("log")
    ax.set_xlim(6.6, 72000)
    ax.set_ylim(*y_limits(rows, metric))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.plot(
        [row["budget"] for row in topk_rows],
        [row[metric] for row in topk_rows],
        color=STYLE["topk"],
        linewidth=2.2,
        alpha=0.35,
        zorder=1,
    )

    for row in rows:
        style = METHOD_STYLE[row["method"]]
        ax.scatter(
            row["budget"],
            row[metric],
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolor="none",
            linewidth=0,
            zorder=4,
        )

    ax.axhline(best[metric], color=STYLE["head"], linewidth=1.0, alpha=0.32, linestyle=(0, (4, 4)), zorder=0)
    ax.text(
        0.985,
        best[metric],
        f"best: {best[metric]:.3f}" if metric == "nll" else f"best: {best[metric]:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=STYLE["head"],
        fontweight="bold",
    )

    ax.set_title(
        f"Validation {label}",
        fontsize=17,
        color=STYLE["ink"],
        fontweight="bold",
        loc="center",
        pad=14,
    )

    ax.set_xlabel("Distillation budget (log scale)", color=STYLE["ink"], labelpad=9)
    ax.set_ylabel(label, color=STYLE["ink"], labelpad=9)
    ticks = [8, 16, 32, 100, 1000, 10000, 50277]
    ax.set_xticks(ticks)
    ax.set_xticklabels([compact_budget(tick) for tick in ticks])

    legend_methods = ["Full KD", "Top-K", "Sampling KD", "Adapt. Top-K", "+ Head-Mass Wt."]
    legend_handles = []
    for method in legend_methods:
        style = {"color": STYLE["topk"], "marker": "o"} if method == "Top-K" else METHOD_STYLE[method]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="none",
                markerfacecolor=style["color"],
                markeredgecolor="none",
                markeredgewidth=0,
                markersize=8.0,
                label=LEGEND_LABELS[method],
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        title="KD Method",
        loc="center left",
        bbox_to_anchor=(1.025, 0.5),
        frameon=True,
        borderpad=0.9,
        handlelength=1.5,
        handletextpad=0.75,
        labelspacing=0.78,
        title_fontproperties={"weight": "bold", "size": 10.6},
    )
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#d5d9df")
    legend.get_frame().set_alpha(0.96)
    legend.get_title().set_color(STYLE["ink"])
    legend.get_title().set_ha("left")
    legend._legend_box.align = "left"

    fig.subplots_adjust(left=0.12, right=0.675, top=0.86, bottom=0.16)
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    setup_matplotlib()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = build_plot_rows(LOG_FILE)
    outputs = [
        plot_metric(rows, "ppl", "PPL", OUTPUT_DIR, "validation_ppl_vs_budget.png"),
        plot_metric(rows, "nll", "NLL", OUTPUT_DIR, "validation_nll_vs_budget.png"),
    ]

    print(f"Read {LOG_FILE}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
