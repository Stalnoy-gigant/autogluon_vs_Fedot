import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AG_COLOR = "#1565C0"
FEDOT_COLOR = "#E65100"
BAR_WIDTH = 0.35


def _paired_bars(ax, labels, ag_values, fedot_values, fmt):
    x = np.arange(len(labels))
    ax.bar(x - BAR_WIDTH / 2, ag_values, BAR_WIDTH, label="AutoGluon", color=AG_COLOR)
    ax.bar(x + BAR_WIDTH / 2, fedot_values, BAR_WIDTH, label="FEDOT", color=FEDOT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2, fontsize=8)


def plot_comparison(results: dict, time_limit: int, path) -> None:
    """results: {'classification': {'autogluon': {...}, 'fedot': {...}}, 'regression': {...}}"""
    clf, reg = results["classification"], results["regression"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    clf_metrics = ["accuracy", "f1", "roc_auc"]
    values = {fw: [clf[fw][m] for m in clf_metrics] for fw in ("autogluon", "fedot")}
    _paired_bars(axes[0], ["Accuracy", "F1", "ROC-AUC"], values["autogluon"], values["fedot"], "%.3f")
    axes[0].set_ylim(min(values["autogluon"] + values["fedot"]) - 0.05, 1.05)
    axes[0].set_title("Классификация (Breast Cancer)")
    axes[0].legend()

    reg_metrics = ["mae", "rmse"]
    _paired_bars(axes[1], ["MAE", "RMSE"],
                 [reg["autogluon"][m] for m in reg_metrics],
                 [reg["fedot"][m] for m in reg_metrics], "%.3f")
    axes[1].set_title("Регрессия, ошибки (ниже лучше)")
    axes[1].legend()

    _paired_bars(axes[2], ["Классификация", "Регрессия"],
                 [clf["autogluon"]["time"], reg["autogluon"]["time"]],
                 [clf["fedot"]["time"], reg["fedot"]["time"]], "%.0f")
    axes[2].axhline(time_limit, color="gray", linestyle="--", linewidth=1, label=f"Лимит {time_limit} с")
    axes[2].set_title("Время обучения, с")
    axes[2].legend()

    fig.suptitle("AutoGluon vs FEDOT", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
