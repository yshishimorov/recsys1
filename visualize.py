import os
import json
import matplotlib.pyplot as plt
import numpy as np
from config import RESULTS_DIR


def plot_comparison():
    results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_metrics.json")]

    test_results = {}
    for fname in results_files:
        if "test" in fname:
            model_name = fname.replace("_metrics.json", "").replace("_test", "")
            with open(os.path.join(RESULTS_DIR, fname), "r") as f:
                test_results[model_name] = json.load(f)

    if len(test_results) < 2:
        print("Нужно минимум 2 модели для сравнения")
        return

    model_names = list(test_results.keys())
    metrics_names = list(test_results[model_names[0]].keys())

    fig, axes = plt.subplots(1, len(metrics_names), figsize=(5 * len(metrics_names), 5))

    if len(metrics_names) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_names):
        values = [test_results[m].get(metric, 0) for m in model_names]
        bars = axes[i].bar(model_names, values, color=["#2196F3", "#FF9800"])
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_ylabel("Score")

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График сохранён: {save_path}")
    plt.close()


def plot_feature_importance(model_path, feature_names, top_n=20):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(model_path)

    importances = model.get_feature_importance()

    if len(feature_names) != len(importances):
        print("Несовпадение числа признаков и важностей")
        return

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_n]

    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#4CAF50")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importances")

    plt.tight_layout()

    model_name = os.path.basename(model_path).replace(".cbm", "")
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"График важности сохранён: {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_comparison()
