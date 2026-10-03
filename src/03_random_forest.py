"""Step 3 — Random Forest (baseline, RandomizedSearchCV, threshold tuning).

Pipeline (as in the report):
  1. Baseline RandomForest with 1,100 trees.
  2. RandomizedSearchCV (10 combinations, 3-fold CV, ROC-AUC).
  3. Decision threshold chosen on 5-fold out-of-fold probabilities: the one with the highest
     precision among those that reach >= 80 % recall.
  4. Evaluation on the test set + feature importances.

Run:   python src/03_random_forest.py          # full pipeline
       python src/03_random_forest.py --fast   # skip baseline + search, use the reported best params
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from common import (
    RANDOM_STATE, RESULTS_DIR, evaluate, load_split, print_metrics, save_figure, save_metrics,
)

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None, "balanced"],
}
# Best parameters reported in the paper (used by --fast)
REPORTED_BEST_PARAMS = {
    "n_estimators": 400, "min_samples_split": 2, "min_samples_leaf": 10,
    "max_depth": 10, "class_weight": "balanced",
}
TARGET_RECALL = 0.80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip the baseline and the random search; use the reported best params")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split()

    if args.fast:
        best_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **REPORTED_BEST_PARAMS)
        best_rf.fit(X_train, y_train)
        best_params = REPORTED_BEST_PARAMS
    else:
        # ── baseline ───────────────────────────────────────────────────────
        rf = RandomForestClassifier(n_estimators=1100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("===== Baseline Random Forest (1100 trees, default threshold) =====")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        # ── randomized search ──────────────────────────────────────────────
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_distributions=PARAM_DISTRIBUTIONS, n_iter=10, cv=3,
            scoring="roc_auc", n_jobs=-1, verbose=3, random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)
        best_rf, best_params = search.best_estimator_, search.best_params_
    print("Best parameters:", best_params)

    # ── threshold tuning on out-of-fold probabilities ──────────────────────
    y_scores_cv = cross_val_predict(
        best_rf, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_cv)
    valid_idx = np.where(recalls[:-1] >= TARGET_RECALL)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No threshold achieves recall >= {TARGET_RECALL}")
    best_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]
    chosen_threshold = float(thresholds[best_idx])
    print(f"Chosen threshold: {chosen_threshold}")

    # ── test performance ───────────────────────────────────────────────────
    y_scores_test = best_rf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_scores_test >= chosen_threshold).astype(int)
    print(classification_report(y_test, y_pred_test))
    print("AUC:", roc_auc_score(y_test, y_scores_test))

    metrics = evaluate(y_test, y_pred_test, y_scores_test)
    print_metrics("Tuned Random Forest — TEST", metrics)
    save_metrics("random_forest_tuned", {**metrics, "threshold": chosen_threshold,
                                         "params": best_params})

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Pastel1_r", colorbar=False)
    for i in range(2):
        for j in range(2):
            disp.text_[i, j].set_color("black")
            disp.text_[i, j].set_text(f"{cm[i, j]:,}")
    plt.title("Confusion Matrix — Tuned Random Forest")
    save_figure("rf_confusion_tuned")

    # ── feature importance ─────────────────────────────────────────────────
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns, "Importance": best_rf.feature_importances_,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    print(feature_importance.iloc[:10])
    feature_importance.to_csv(RESULTS_DIR / "rf_feature_importance.csv", index=False)


if __name__ == "__main__":
    main()
