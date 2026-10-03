"""Step 2 — Logistic Regression (baseline, balanced and tuned) + SHAP.

Three models, as in the report:
  1. Normal logistic regression
  2. Balanced logistic regression (class_weight="balanced")
  3. Tuned logistic regression: grid over C, penalty/solver, class_weight and decision
     threshold (6 configs x 11 C values x 91 thresholds = 6,006 combinations), evaluated on a
     validation split carved out of the training set. Selection rule: highest recall among
     models with validation accuracy >= 0.70.

Run:  python src/02_logistic_regression.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import (
    RANDOM_STATE, RESULTS_DIR, evaluate, kernel_shap_values, load_split,
    print_metrics, save_figure, save_metrics,
)

C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
THRESHOLD_VALUES = np.round(np.arange(0.05, 0.96, 0.01), 2)
CONFIGS = [
    {"penalty": "l2", "solver": "lbfgs", "class_weight": None},
    {"penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
    {"penalty": "l1", "solver": "liblinear", "class_weight": None},
    {"penalty": "l1", "solver": "liblinear", "class_weight": "balanced"},
    {"penalty": "l2", "solver": "liblinear", "class_weight": None},
    {"penalty": "l2", "solver": "liblinear", "class_weight": "balanced"},
]
MIN_ACCURACY = 0.70
# NOTE: the size of the validation split is not shown in the report; 20 % is assumed.
VAL_SIZE = 0.2


def run_baselines(X_train, X_test, y_train, y_test) -> None:
    """Normal and balanced logistic regression on the standardised full training set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, extra in [("normal", {}), ("balanced", {"class_weight": "balanced"})]:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **extra)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        print(f"\n===== {name.capitalize()} Logistic Regression =====")
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        print(classification_report(y_test, y_pred))
        save_metrics(f"logistic_regression_{name}", evaluate(y_test, y_pred, y_proba))

        ConfusionMatrixDisplay.from_estimator(
            model, X_test_scaled, y_test, display_labels=["No Diabetes", "Diabetes"]
        )
        plt.title(f"Confusion Matrix on {name.capitalize()} Logistic Regression")
        save_figure(f"lr_confusion_{name}")


def tune_logistic_regression(X_train, y_train):
    """Grid over (config, C, threshold) on a validation split; returns the best model."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    yv = np.asarray(y_val)

    rows, models = [], {}
    for config_id, config in enumerate(CONFIGS):
        for C in C_VALUES:
            model = LogisticRegression(
                C=C, penalty=config["penalty"], solver=config["solver"],
                class_weight=config["class_weight"], max_iter=3000,
                random_state=RANDOM_STATE,
            )
            model.fit(X_tr_scaled, y_tr)
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            roc_auc = roc_auc_score(yv, val_proba)
            model_name = (f"LR_penalty={config['penalty']}_solver={config['solver']}_"
                          f"class_weight={config['class_weight']}_C={C}")
            models[model_name] = model

            for threshold in THRESHOLD_VALUES:
                pred = (val_proba >= threshold).astype(int)
                m = evaluate(yv, pred)
                rows.append({
                    "model_name": model_name, "config_id": config_id, **config,
                    "C": C, "alpha": 1 / C, "threshold": threshold,
                    "accuracy": m["accuracy"], "precision": m["precision"],
                    "recall": m["recall"], "f1_score": m["f1"], "roc_auc": roc_auc,
                    "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                    "specificity_no_diabetes": m["tn"] / (m["tn"] + m["fp"]),
                    "false_negative_rate": m["fn"] / (m["fn"] + m["tp"]),
                    "false_positive_rate": m["fp"] / (m["fp"] + m["tn"]),
                })
        print(f"  config {config_id + 1}/{len(CONFIGS)} done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "lr_tuning_results.csv", index=False)

    valid = results_df[results_df["accuracy"] >= MIN_ACCURACY].copy()
    print("Number of combinations tested:", len(results_df))
    print(f"Combinations with validation accuracy >= {MIN_ACCURACY}:", len(valid))
    if valid.empty:
        raise ValueError("No model reached the minimum validation accuracy. "
                         "Lower MIN_ACCURACY or expand the search.")

    ranked = valid.sort_values(
        by=["recall", "fn", "f1_score", "precision", "roc_auc", "accuracy"],
        ascending=[False, True, False, False, False, False],
    )
    best_row = ranked.iloc[0]
    return {
        "model": models[best_row["model_name"]],
        "scaler": scaler,
        "X_train_model_scaled": X_tr_scaled,
        "threshold": float(best_row["threshold"]),
        "best_row": best_row,
    }


def main() -> None:
    X_train, X_test, y_train, y_test = load_split()

    run_baselines(X_train, X_test, y_train, y_test)

    print("\n===== Tuned Logistic Regression (this takes a few minutes) =====")
    tuned = tune_logistic_regression(X_train, y_train)
    best_model, best_threshold = tuned["model"], tuned["threshold"]
    best_row = tuned["best_row"]
    print("Best model    :", best_row["model_name"])
    print("Best threshold:", best_threshold)
    print("\nValidation results of the best combination:")
    print(best_row)

    X_test_scaled = tuned["scaler"].transform(X_test)
    test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    metrics = evaluate(y_test, test_pred, test_proba)
    print_metrics("Tuned Logistic Regression — TEST", metrics)
    print(classification_report(y_test, test_pred,
                                target_names=["No Diabetes", "Diabetes"], zero_division=0))
    save_metrics("logistic_regression_tuned", {
        **metrics, "threshold": best_threshold,
        "params": {"penalty": best_row["penalty"], "solver": best_row["solver"],
                   "class_weight": best_row["class_weight"], "C": best_row["C"]},
    })

    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=["No Diabetes", "Diabetes"]
    )
    plt.title("Confusion Matrix on Tuned Logistic Regression")
    save_figure("lr_confusion_tuned")

    # ── SHAP (KernelExplainer on 100 test observations) ────────────────────
    print("\nComputing SHAP values (KernelExplainer, a few minutes)...")
    X_test_sample = X_test_scaled[:100]
    shap_vals, _ = kernel_shap_values(
        best_model.predict_proba, tuned["X_train_model_scaled"], X_test_sample
    )
    feature_names = list(X_train.columns)

    shap.summary_plot(shap_vals, X_test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot — Tuned Logistic Regression (Diabetes = 1)", fontweight="bold")
    save_figure("lr_shap_summary")

    shap.summary_plot(shap_vals, X_test_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontweight="bold")
    save_figure("lr_shap_bar")


if __name__ == "__main__":
    main()
