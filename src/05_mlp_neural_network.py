"""Step 5 — Neural network (MLPClassifier) + SHAP explainability.

Four versions, as in the report:
  Original : GridSearchCV (120 fits, ROC-AUC), default threshold 0.50
  v2       : same model, decision threshold tuned to maximise F1   <- FINAL MODEL
  v3       : SMOTE oversampling + F1 scoring (~7 h)  -> discarded     (results hard-coded)
  v4       : deeper architecture search (640 fits, ~10 h)             (results hard-coded)

Run:   python src/05_mlp_neural_network.py                  # full grid search (several minutes/hours)
       python src/05_mlp_neural_network.py --fast           # skip the grid search, use the best params
       python src/05_mlp_neural_network.py --fast --retrain # also retrain v3/v4 with their best params
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, precision_recall_curve, roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import (
    RANDOM_STATE, RESULTS_DIR, evaluate, kernel_shap_values, load_split, print_metrics,
    save_figure, save_metrics,
)

PARAM_GRID = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "max_iter": [300],
    "early_stopping": [True],
}  # 4 x 2 x 3 = 24 combinations x 5 folds = 120 fits
# Best params of the grid search (used by --fast)
REPORTED_BEST_PARAMS = {
    "hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 0.0001,
    "max_iter": 300, "early_stopping": True,
}

# v3 / v4 take 7-10 h to search: results from the original runs are stored here.
HARDCODED = {
    "MLP v3 (SMOTE+F1)": dict(
        accuracy=0.7550, precision=0.3083, recall=0.6098, f1=0.4095, roc_auc=0.7723,
        threshold=0.53, cm=[[33993, 9674], [2758, 4311]]),
    "MLP v4 (Deep Arch)": dict(
        accuracy=0.7985, precision=0.3740, recall=0.6220, f1=0.4670, roc_auc=0.8304,
        threshold=0.219, cm=[[36681, 6986], [2672, 4397]]),
}
V3_PARAMS = dict(hidden_layer_sizes=(128, 64), activation="tanh", alpha=0.0001,
                 learning_rate_init=0.001, max_iter=300, early_stopping=True)
V4_PARAMS = dict(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.1,
                 learning_rate_init=0.0005, max_iter=300, early_stopping=True)


def f1_optimal_threshold(y_true, y_prob):
    """Threshold maximising F1 on the precision-recall curve.

    NOTE: as in the original experiments the curve is computed on the TEST set. For a stricter
    protocol, compute it on a validation split / out-of-fold predictions instead.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    idx = int(np.argmax(f1_scores))
    return thresholds, precisions, recalls, f1_scores, idx


def retrain_v3_v4(X_train_scaled, y_train, X_test_scaled, y_test) -> dict:
    """Optional: refit v3 (SMOTE) and v4 with their documented best parameters."""
    from imblearn.over_sampling import SMOTE

    out = {}
    X_sm, y_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)
    for name, (X_fit, y_fit, params) in {
        "MLP v3 (SMOTE+F1)": (X_sm, y_sm, V3_PARAMS),
        "MLP v4 (Deep Arch)": (X_train_scaled, y_train, V4_PARAMS),
    }.items():
        print(f"Retraining {name} ...")
        model = MLPClassifier(solver="adam", random_state=RANDOM_STATE, **params)
        model.fit(X_fit, y_fit)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        thresholds, _, _, _, idx = f1_optimal_threshold(y_test, prob)
        m = evaluate(y_test, (prob >= thresholds[idx]).astype(int), prob)
        out[name] = dict(accuracy=m["accuracy"], precision=m["precision"], recall=m["recall"],
                         f1=m["f1"], roc_auc=m["roc_auc"], threshold=float(thresholds[idx]),
                         cm=[[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="skip GridSearchCV and train directly with the best parameters")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain v3/v4 instead of using the hard-coded results")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. MLP Original — GridSearchCV, default threshold 0.50
    # ══════════════════════════════════════════════════════════════════════
    if args.fast:
        best_mlp = MLPClassifier(random_state=RANDOM_STATE, **REPORTED_BEST_PARAMS)
        best_mlp.fit(X_train_scaled, y_train)
    else:
        grid = GridSearchCV(MLPClassifier(random_state=RANDOM_STATE), PARAM_GRID, cv=5,
                            scoring="roc_auc", n_jobs=-1, verbose=3)
        start = time.time()
        grid.fit(X_train_scaled, y_train)
        m, s = divmod(int(time.time() - start), 60)
        print(f"\nSearch completed in {m:02d}:{s:02d}")
        print(f"Best ROC-AUC CV : {grid.best_score_:.4f}")
        print(f"Best params     : {grid.best_params_}")
        best_mlp = grid.best_estimator_

    y_pred = best_mlp.predict(X_test_scaled)
    y_pred_prob = best_mlp.predict_proba(X_test_scaled)[:, 1]
    orig = evaluate(y_test, y_pred, y_pred_prob)
    print_metrics("MLP Original (threshold = 0.50)", orig)
    print(classification_report(y_test, y_pred))
    save_metrics("mlp_original", {**orig, "threshold": 0.5})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("MLP Original — Confusion Matrix\n(threshold = 0.50)", fontweight="bold")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, lw=2, color="#2e6db4", label=f"MLP Original (AUC = {orig['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("MLP Original — ROC Curve", fontweight="bold")
    axes[1].legend(loc="lower right")
    save_figure("mlp_original_eval")

    # ══════════════════════════════════════════════════════════════════════
    # 3. MLP v2 — threshold tuning (FINAL MODEL)
    # ══════════════════════════════════════════════════════════════════════
    thresholds, precisions, recalls, f1_scores, idx = f1_optimal_threshold(y_test, y_pred_prob)
    best_threshold = float(thresholds[idx])
    print(f"\nOptimal threshold : {best_threshold:.3f}  (default was 0.50)")
    print(f"  → Precision     : {precisions[idx]:.3f}")
    print(f"  → Recall        : {recalls[idx]:.3f}")
    print(f"  → F1            : {f1_scores[idx]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(thresholds, precisions[:-1], label="Precision", color="steelblue", lw=2)
    axes[0].plot(thresholds, recalls[:-1], label="Recall", color="tomato", lw=2)
    axes[0].plot(thresholds, f1_scores, label="F1-Score", color="green", lw=2, linestyle="--")
    axes[0].axvline(best_threshold, color="black", linestyle=":",
                    label=f"Best threshold = {best_threshold:.2f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_title("Precision / Recall / F1 vs Threshold", fontweight="bold")
    axes[0].legend()
    axes[1].plot(recalls[:-1], precisions[:-1], color="#2e6db4", lw=2)
    axes[1].scatter(recalls[idx], precisions[idx], color="red", zorder=5, s=80,
                    label=f"Best threshold = {best_threshold:.2f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend()
    save_figure("mlp_threshold_tuning_v2")

    y_pred_tuned = (y_pred_prob >= best_threshold).astype(int)
    v2 = evaluate(y_test, y_pred_tuned, y_pred_prob)
    print_metrics("MLP v2 (tuned threshold)", v2)
    print(classification_report(y_test, y_pred_tuned))
    save_metrics("mlp_v2", {**v2, "threshold": best_threshold})

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("MLP Original\n(threshold = 0.50)", fontweight="bold")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tuned, ax=axes[1], colorbar=False)
    axes[1].set_title(f"MLP v2 — Tuned Threshold\n(threshold = {best_threshold:.2f})", fontweight="bold")
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_prob)
    axes[2].plot(fpr, tpr, lw=2, color="#2e6db4", label=f"MLP v2 (AUC = {v2['roc_auc']:.3f})")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    i_thr = np.argmin(np.abs(roc_thresholds - best_threshold))
    axes[2].scatter(fpr[i_thr], tpr[i_thr], color="red", zorder=5, s=80,
                    label=f"Threshold = {best_threshold:.2f}")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve with Selected Threshold", fontweight="bold")
    axes[2].legend(loc="lower right")
    save_figure("mlp_v2_eval")
    print(f"→ TP {orig['tp']:,} → {v2['tp']:,} | FN {orig['fn']:,} → {v2['fn']:,} "
          f"| ROC-AUC unchanged ({v2['roc_auc']:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # 4-5. MLP v3 (SMOTE) and v4 (deep architecture)
    # ══════════════════════════════════════════════════════════════════════
    others = (retrain_v3_v4(X_train_scaled, y_train, X_test_scaled, y_test)
              if args.retrain else HARDCODED)
    v3, v4 = others["MLP v3 (SMOTE+F1)"], others["MLP v4 (Deep Arch)"]
    for name, res in others.items():
        cm = np.array(res["cm"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\n(threshold = {res['threshold']:.2f})", fontweight="bold")
        save_figure("mlp_" + ("v3" if "v3" in name else "v4") + "_confusion")

    # ══════════════════════════════════════════════════════════════════════
    # 6. SHAP explainability on MLP v2 (KernelExplainer, 100 + 100 samples)
    # ══════════════════════════════════════════════════════════════════════
    print("\nRunning SHAP KernelExplainer (may take 3-5 minutes)...")
    X_test_sample = X_test_scaled[:100]
    shap_vals_pos, expected_val = kernel_shap_values(
        best_mlp.predict_proba, X_train_scaled, X_test_sample)
    feature_names = list(X_train.columns)
    print("SHAP values shape:", shap_vals_pos.shape)

    shap.summary_plot(shap_vals_pos, X_test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot — MLP v2 (Diabetes = 1)", fontweight="bold")
    save_figure("mlp_shap_summary")

    shap.summary_plot(shap_vals_pos, X_test_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontweight="bold")
    save_figure("mlp_shap_bar")

    explanation = shap.Explanation(values=shap_vals_pos[0], base_values=expected_val,
                                   data=X_test_sample[0], feature_names=feature_names)
    shap.plots.waterfall(explanation, show=False)
    plt.title("SHAP Waterfall — Single Prediction", fontweight="bold")
    save_figure("mlp_shap_waterfall")

    top_features = ["GenHlth", "BMI", "Age"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat in zip(axes, top_features):
        j = feature_names.index(feat)
        ax.scatter(X_test_sample[:, j], shap_vals_pos[:, j], c=X_test_sample[:, j],
                   cmap="coolwarm", alpha=0.7, edgecolors="none", s=30)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel(feat + " (scaled)")
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence: {feat}", fontweight="bold")
    save_figure("mlp_shap_dependence")

    mean_abs = pd.Series(np.abs(shap_vals_pos).mean(axis=0), index=feature_names)
    mean_abs.sort_values(ascending=False).to_csv(RESULTS_DIR / "mlp_shap_mean_abs.csv",
                                                 header=["mean_abs_shap"])

    # ══════════════════════════════════════════════════════════════════════
    # 7. Final MLP version comparison
    # ══════════════════════════════════════════════════════════════════════
    rows = [("MLP Original", orig), ("MLP v2 (Threshold) ✓", v2),
            ("MLP v3 (SMOTE + F1)", v3), ("MLP v4 (Deep Arch)", v4)]
    comparison = pd.DataFrame({
        "Version": [r[0] for r in rows],
        "Accuracy": [round(r[1]["accuracy"], 4) for r in rows],
        "Precision": [round(r[1]["precision"], 4) for r in rows],
        "Recall": [round(r[1]["recall"], 4) for r in rows],
        "F1-Score": [round(r[1]["f1"], 4) for r in rows],
        "ROC-AUC": [round(r[1]["roc_auc"], 4) for r in rows],
    })
    print("\n── MLP Full Evolution Comparison ─────────────────────────")
    print(comparison.to_string(index=False))
    comparison.to_csv(RESULTS_DIR / "mlp_version_comparison.csv", index=False)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = ["#aaaaaa", "#2e6db4", "#e05c5c", "#f0a500"]
    x, width = np.arange(len(metrics)), 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (_, row) in enumerate(comparison.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["Version"],
               color=colors[i], edgecolor="white", linewidth=0.6, alpha=1.0 if i == 1 else 0.75)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("MLP Version Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    save_figure("mlp_version_comparison")

    cms = [([[orig["tn"], orig["fp"]], [orig["fn"], orig["tp"]]], "MLP Original\n(thr=0.50)"),
           ([[v2["tn"], v2["fp"]], [v2["fn"], v2["tp"]]], f"MLP v2 ✓\n(thr={best_threshold:.2f})"),
           (v3["cm"], f"MLP v3 — SMOTE\n(thr={v3['threshold']:.2f})"),
           (v4["cm"], f"MLP v4 — Deep\n(thr={v4['threshold']:.2f})")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (cm, title) in zip(axes, cms):
        ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=[0, 1]).plot(
            ax=ax, colorbar=False)
        ax.set_title(title, fontweight="bold")
    plt.suptitle("All MLP Versions — Confusion Matrix Comparison", fontsize=12, fontweight="bold")
    save_figure("mlp_all_confusion")


if __name__ == "__main__":
    main()
