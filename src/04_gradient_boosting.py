"""Step 4 — Gradient Boosting with LightGBM (three variants) + SHAP.

Variants (as in the report):
  custom   : asymmetric, cost-sensitive custom loss (TP +2, FP -2, FN -10)
  binary   : standard 'binary' (logistic) objective          <- best model of the project
  balanced : 'binary' objective + scale_pos_weight = 6.2

For every variant: RandomizedSearchCV (15 combinations, 5-fold CV, ROC-AUC computed from raw
scores), then a decision threshold on the raw scores selected on 5-fold out-of-fold predictions
(highest precision subject to recall >= TARGET_RECALL), then test-set evaluation.

Run:   python src/04_gradient_boosting.py                    # all three variants (long)
       python src/04_gradient_boosting.py --model binary     # only the best model
       python src/04_gradient_boosting.py --model binary --fast  # skip the search, use reported params

GPU: LightGBM runs on CPU by default. If you have a GPU-enabled LightGBM build, set the
environment variable LGBM_DEVICE=gpu before running.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from common import (
    RANDOM_STATE, evaluate, load_split, positive_class_shap, print_metrics, save_figure,
    save_metrics,
)

LGBM_DEVICE = os.environ.get("LGBM_DEVICE", "cpu")
SCALE_POS_WEIGHT = 6.2  # class ratio (non-diabetic / diabetic)
TARGET_RECALL = 0.814   # value used in the code (the report text mentions "at least 80 %")

PARAMETER_GRID = {
    "n_estimators": [300, 500, 1000],
    "learning_rate": [0.005, 0.01, 0.02, 0.05],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [20, 50, 100, 200],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l1": [0, 0.1, 1, 5],
    "lambda_l2": [0, 0.1, 1, 5],
}
# Best parameters reported for the 'binary' model (used by --fast for every variant)
REPORTED_BEST_PARAMS = {
    "num_leaves": 15, "n_estimators": 500, "min_child_samples": 20, "learning_rate": 0.05,
    "lambda_l2": 1, "lambda_l1": 1, "feature_fraction": 0.6, "bagging_fraction": 0.8,
}


# ----------------------------------------------------------------------------- objectives
def asymmetric_loss(y_real, y_pred):
    """Cost-sensitive binary cross-entropy gradient/Hessian (custom LightGBM objective).

    Positive-class weight = C_FN - C_TP = 10 - 2 = 8, negative-class weight = C_FP = 2.
    """
    false_negative = 10.0
    false_positive = 2.0
    true_positive = 2.0
    probabilities = 1.0 / (1.0 + np.exp(-y_pred))
    pos_weight = false_negative - true_positive
    gradient = (probabilities * (false_positive * (1 - y_real) + y_real * pos_weight)
                - pos_weight * y_real)
    hessian = probabilities * (1.0 - probabilities) * (
        false_positive * (1.0 - y_real) + y_real * pos_weight)
    hessian = np.maximum(hessian, 1e-4)
    return gradient, hessian


def raw_score_roc_auc(estimator, X, y):
    """Scorer for RandomizedSearchCV: custom objectives return raw scores, not probabilities."""
    return roc_auc_score(y, estimator.predict(X, raw_score=True))


def build_model(variant: str, params: dict | None = None) -> LGBMClassifier:
    kwargs = dict(random_state=RANDOM_STATE, verbose=-1, device=LGBM_DEVICE, n_jobs=-1)
    kwargs["objective"] = asymmetric_loss if variant == "custom" else "binary"
    if variant == "balanced":
        kwargs["scale_pos_weight"] = SCALE_POS_WEIGHT
    if params:
        kwargs.update(params)
    return LGBMClassifier(**kwargs)


# ----------------------------------------------------------------------------- pipeline
def select_threshold(variant, best_params, X_train, y_train):
    """Threshold on raw scores from 5-fold OOF predictions (max precision s.t. recall target)."""
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.zeros(len(X_train))
    for train_idx, validation_idx in kfolds.split(X_train, y_train):
        model = build_model(variant, best_params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_scores[validation_idx] = model.predict(X_train.iloc[validation_idx], raw_score=True)

    possible_thresholds = np.percentile(oof_scores, np.linspace(0.1, 99.9, 1000))
    positives = np.asarray(y_train) == 1
    best_threshold, best_precision = None, -1.0
    for threshold in possible_thresholds:
        pred = oof_scores > threshold
        tp = np.sum(pred & positives)
        if pred.sum() == 0:
            continue
        precision, recall = tp / pred.sum(), tp / positives.sum()
        if recall >= TARGET_RECALL and precision > best_precision:
            best_precision, best_threshold = precision, float(threshold)
    if best_threshold is None:
        raise ValueError(f"No threshold achieves recall >= {TARGET_RECALL}")
    return best_threshold


def run_variant(variant, X_train, X_test, y_train, y_test, fast):
    print(f"\n{'=' * 70}\nGradient Boosting — variant: {variant}\n{'=' * 70}")
    if fast:
        best_params = dict(REPORTED_BEST_PARAMS)
        best_model = build_model(variant, best_params).fit(X_train, y_train)
    else:
        search = RandomizedSearchCV(
            estimator=build_model(variant), param_distributions=PARAMETER_GRID, n_iter=15, cv=5,
            scoring=raw_score_roc_auc, random_state=RANDOM_STATE, verbose=3, n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_params, best_model = search.best_params_, search.best_estimator_
        print("CV AUC:", search.best_score_)
    print("Best parameters:", best_params)

    threshold = select_threshold(variant, best_params, X_train, y_train)
    print("Best threshold (raw score):", threshold)

    test_scores = best_model.predict(X_test, raw_score=True)
    y_pred = (test_scores > threshold).astype(int)
    metrics = evaluate(y_test, y_pred, test_scores)
    print_metrics(f"Gradient Boosting [{variant}] — TEST", metrics)
    print(classification_report(y_test, y_pred))
    save_metrics(f"gradient_boosting_{variant}",
                 {**metrics, "threshold": threshold, "params": best_params})

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Pastel1_r", colorbar=False)
    for i in range(2):
        for j in range(2):
            disp.text_[i, j].set_color("black")
            disp.text_[i, j].set_text(f"{cm[i, j]:,}")
    plt.title(f"Confusion Matrix — LightGBM ({variant})")
    save_figure(f"gb_confusion_{variant}")
    return best_model


def shap_analysis(model, X_test) -> None:
    """TreeExplainer on a random sample of 5,000 test rows."""
    sample = X_test.sample(min(5000, len(X_test)), random_state=RANDOM_STATE)
    shap_values = positive_class_shap(shap.TreeExplainer(model).shap_values(sample))

    shap.summary_plot(shap_values, sample, show=False)
    save_figure("gb_shap_summary")
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    save_figure("gb_shap_bar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "binary", "balanced", "all"], default="all")
    parser.add_argument("--fast", action="store_true",
                        help="skip RandomizedSearchCV and use the reported best parameters")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split()
    variants = ["custom", "binary", "balanced"] if args.model == "all" else [args.model]

    for variant in variants:
        model = run_variant(variant, X_train, X_test, y_train, y_test, args.fast)
        if variant == "binary":  # SHAP is reported for the best model (Model 2)
            shap_analysis(model, X_test)


if __name__ == "__main__":
    main()
