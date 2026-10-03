"""Step 6 — Cross-model comparison (Table 5 of the report).

Uses the metrics saved in ``results/*.json`` by scripts 02-05 when they exist; otherwise falls
back to the values reported in the paper (test set, 50,736 patients).

Run:  python src/06_cross_model_comparison.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR, save_figure

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

# (label, results file stem, values reported in the paper)
MODELS = [
    ("Logistic Regression", "logistic_regression_tuned", [0.7036, 0.2948, 0.8099, 0.4322, 0.8197]),
    ("Random Forest", "random_forest_tuned", [0.71, 0.30, 0.80, 0.44, 0.822]),
    ("Gradient Boosting", "gradient_boosting_binary", [0.71, 0.30, 0.81, 0.44, 0.82789]),
    ("MLP v2", "mlp_v2", [0.7985, 0.3693, 0.6305, 0.4658, 0.8261]),
]


def load_values(stem, fallback):
    path = RESULTS_DIR / f"{stem}_metrics.json"
    if not path.exists():
        return fallback, "paper"
    m = json.loads(path.read_text(encoding="utf-8"))
    return [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]], "this run"


def main() -> None:
    rows, sources = [], []
    for label, stem, fallback in MODELS:
        values, source = load_values(stem, fallback)
        rows.append([label, *values])
        sources.append(source)
    table = pd.DataFrame(rows, columns=["Model", *METRICS])
    print(table.round(4).to_string(index=False))
    print("\nSource of each row:", dict(zip(table["Model"], sources)))
    table.round(4).to_csv(RESULTS_DIR / "cross_model_comparison.csv", index=False)

    # ── grouped bar chart ──────────────────────────────────────────────────
    colors = ["#e07b39", "#8c564b", "#2ca02c", "#2e6db4"]
    x, width = np.arange(len(METRICS)), 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, row in table.iterrows():
        ax.bar(x + i * width, row[METRICS].astype(float), width, label=row["Model"],
               color=colors[i], edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(METRICS, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("All Models — Final Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    save_figure("all_models_comparison")

    # ── recall: the primary metric (clinical goal = minimise missed diagnoses) ─
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in table.iterrows():
        ax.bar(i, row["Recall"], color=colors[i], edgecolor="white", width=0.5)
        ax.text(i, row["Recall"] + 0.01, f"{row['Recall']:.3f}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([m.replace(" ", "\n", 1) for m in table["Model"]], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title("Recall Comparison — All Models\n(% of diabetic patients correctly detected)",
                 fontsize=12, fontweight="bold")
    save_figure("recall_comparison")

    print("\n→ Best model overall: Gradient Boosting (binary objective) — highest recall and AUC.")
    print("→ Top-5 SHAP features in every model: GenHlth, BMI, Age, HighBP, HighChol.")


if __name__ == "__main__":
    main()
