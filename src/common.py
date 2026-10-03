"""Shared utilities for the Diabetes Risk Prediction project.

Replaces the Google Colab / Google Drive setup of the original notebooks:
* paths are relative to the repository root,
* the post-processed Excel in the repo root is the data source (cached as CSV in ``data/``);
  without it, the dataset is downloaded from the UCI repository and split the same way,
* figures are saved to ``figures/`` and metrics to ``results/`` (no blocking windows
  unless the environment variable ``SHOW_PLOTS=1`` is set).
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")  # headless backend: figures are only written to disk

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "cdc_diabetes_health_indicators.csv"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

# Post-processed split shipped with the repository (sheets "train" and "test"), kept in the repo root.
EXCEL_NAME = "diabetes_train_test_split.xlsx"

TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.2
UCI_DATASET_ID = 891  # CDC Diabetes Health Indicators


# ----------------------------------------------------------------------------- data
def load_raw_dataset(force: bool = False) -> pd.DataFrame:
    """Return the full CDC dataset (features + target), downloading it on first use."""
    if RAW_CSV.exists() and not force:
        return pd.read_csv(RAW_CSV)

    from ucimlrepo import fetch_ucirepo

    print("Downloading CDC Diabetes Health Indicators from the UCI repository...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df.to_csv(RAW_CSV, index=False)
    return df


def make_split(df: pd.DataFrame):
    """Stratified 80/20 split with random_state=42 (as described in the report)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def save_split(X_train, X_test, y_train, y_test) -> None:
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_CSV, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)


def find_excel() -> Path | None:
    """Locate the post-processed Excel in the repository root (or None)."""
    exact = ROOT / EXCEL_NAME
    if exact.exists():
        return exact
    candidates = sorted(ROOT.glob("*diabetes*.xlsx"))
    return candidates[0] if candidates else None


def excel_to_csv_cache(excel_path: Path) -> None:
    """Read the Excel once (slow: ~250k rows) and cache both sheets as CSV."""
    print(f"Reading {excel_path.name} (first run only, this can take a minute)...")
    pd.read_excel(excel_path, sheet_name="train").to_csv(TRAIN_CSV, index=False)
    pd.read_excel(excel_path, sheet_name="test").to_csv(TEST_CSV, index=False)


def load_split():
    """Return ``X_train, X_test, y_train, y_test``.

    Priority:
      1. ``diabetes_train_test_split.xlsx`` in the repo root (sheets "train" / "test"),
         cached as CSV in ``data/processed`` and refreshed if the Excel is newer.
      2. If there is no Excel: download the dataset from UCI and build the same
         stratified 80/20 split (random_state=42).
    """
    excel = find_excel()
    if excel is not None:
        stale = (not TRAIN_CSV.exists() or not TEST_CSV.exists()
                 or excel.stat().st_mtime > min(TRAIN_CSV.stat().st_mtime, TEST_CSV.stat().st_mtime))
        if stale:
            excel_to_csv_cache(excel)
    elif not (TRAIN_CSV.exists() and TEST_CSV.exists()):
        X_train, X_test, y_train, y_test = make_split(load_raw_dataset())
        save_split(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return (
        train.drop(columns=[TARGET]),
        test.drop(columns=[TARGET]),
        train[TARGET],
        test[TARGET],
    )


# ----------------------------------------------------------------------------- metrics
def evaluate(y_true, y_pred, y_score=None) -> dict:
    """Binary-classification metrics for the positive class (diabetes = 1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def print_metrics(title: str, m: dict) -> None:
    print(f"\n── {title} " + "─" * max(0, 56 - len(title)))
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in m:
            print(f"{key:<10}: {m[key]:.4f}")
    print(f"confusion : TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TP={m['tp']:,}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_metrics(name: str, payload: dict) -> Path:
    path = RESULTS_DIR / f"{name}_metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- plots
def save_figure(name: str) -> Path:
    """Save the current matplotlib figure to ``figures/<name>.png``."""
    plt.tight_layout()
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    return path


# ----------------------------------------------------------------------------- SHAP
def positive_class_shap(shap_values):
    """Normalise the different SHAP output formats to a 2-D array for class 1."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if getattr(shap_values, "ndim", 2) == 3:
        return shap_values[:, :, 1]
    return shap_values


def kernel_shap_values(predict_proba, X_background_source, X_sample,
                       n_background: int = 100, nsamples: int = 100):
    """Model-agnostic SHAP (KernelExplainer) on a small sample, as in the report."""
    import shap

    background = shap.sample(X_background_source, n_background, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = expected[1] if len(np.atleast_1d(expected)) > 1 else np.atleast_1d(expected)[0]
    return positive_class_shap(shap_values), expected
