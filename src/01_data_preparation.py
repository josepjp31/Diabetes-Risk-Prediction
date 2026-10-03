"""Step 1 — Data checks and train/test split.

Two ways to get the data (the first one is the default):

  A) The post-processed Excel ``diabetes_train_test_split.xlsx`` (sheets "train" and "test") sitting
     in the repository root. The script loads it, caches it as CSV and runs the sanity checks.
  B) No Excel: the dataset is downloaded from UCI (``ucimlrepo``, id=891) and the stratified
     80/20 split (random_state=42) is created.

Extra flag:
  --export-excel   download from UCI, build the split and WRITE ``diabetes_train_test_split.xlsx``
                   in the repo root (this is how the Excel is generated).

Run:  python src/01_data_preparation.py [--export-excel]
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    EXCEL_NAME, ROOT, TARGET, find_excel, load_raw_dataset, load_split, make_split,
    save_figure, save_split,
)


def export_excel() -> None:
    """Equivalent to the original Colab step that created the Excel on Google Drive."""
    df = load_raw_dataset(force=True)  # fetch_ucirepo(id=891): features + targets
    X_train, X_test, y_train, y_test = make_split(df)
    save_split(X_train, X_test, y_train, y_test)
    path = ROOT / EXCEL_NAME
    with pd.ExcelWriter(path) as writer:
        pd.concat([X_train, y_train], axis=1).to_excel(writer, sheet_name="train", index=False)
        pd.concat([X_test, y_test], axis=1).to_excel(writer, sheet_name="test", index=False)
    print(f"Excel written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-excel", action="store_true",
                        help="download from UCI and write diabetes_train_test_split.xlsx")
    args = parser.parse_args()

    if args.export_excel:
        export_excel()

    X_train, X_test, y_train, y_test = load_split()
    source = find_excel()
    print("Data source:", source.name if source else "UCI download (no Excel found)")
    print(f"X_train {X_train.shape} | X_test {X_test.shape}")

    full = pd.concat([pd.concat([X_train, y_train], axis=1),
                      pd.concat([X_test, y_test], axis=1)], ignore_index=True)
    print("\nFull dataset shape:", full.shape)
    print(full.describe().T)

    # ── target distribution ────────────────────────────────────────────────
    print("\nTarget proportions (%):")
    print(full[TARGET].value_counts(normalize=True) * 100)
    print("Train:", y_train.value_counts().to_dict(), "| Test:", y_test.value_counts().to_dict())

    # ── duplicates (kept: they are different people with identical answers) ─
    duplicate_count = full.duplicated().sum()
    print("\nDuplicated rows:", duplicate_count, f"({duplicate_count / len(full):.2%})")

    # ── missing values ─────────────────────────────────────────────────────
    print("Total missing values:", full.isnull().sum().sum())

    # ── class-distribution plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in zip(axes, [y_train, y_test], ["Training Set", "Test Set"]):
        counts = y.value_counts().sort_index()
        bars = ax.bar(["No Diabetes (0)", "Diabetes (1)"], counts.values,
                      color=["#2e6db4", "#e05c5c"], edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{count:,}\n({count / len(y) * 100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(counts.values) * 1.18)
    plt.suptitle("Class Distribution — Diabetes Dataset", fontsize=13, fontweight="bold")
    save_figure("class_distribution")
    print("→ Clear class imbalance: ~6:1 ratio (No Diabetes vs Diabetes)")


if __name__ == "__main__":
    main()
