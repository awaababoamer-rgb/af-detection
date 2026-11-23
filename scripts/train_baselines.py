"""Phase 4: train logistic regression and random forest baselines.

Protocol (no leakage, test set touched once):
1. Stratified split: 70% train / 15% validation / 15% test. Stratified so the
   rare AF and Noisy classes appear in every split at the same rate.
2. Every candidate model is fit on train and scored on validation (macro F1).
3. The best candidate is refit on train+validation and evaluated ONCE on test.

Usage:
    python scripts/train_baselines.py
Outputs:
    reports/results.json, reports/validation_scores.csv, reports/test_errors.csv,
    reports/figures/confusion_matrix.png, reports/figures/feature_importance.png,
    models/model.joblib (used by the web demo)
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from af.config import CLASS_ORDER, FEATURES_CSV, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from af.features import FEATURE_COLUMNS
from af.models import evaluate, make_logistic_regression, make_random_forest

CANDIDATES = {
    "logreg_C0.1": lambda: make_logistic_regression(C=0.1),
    "logreg_C1": lambda: make_logistic_regression(C=1.0),
    "logreg_C10": lambda: make_logistic_regression(C=10.0),
    "rf_leaf1": lambda: make_random_forest(min_samples_leaf=1),
    "rf_leaf3": lambda: make_random_forest(min_samples_leaf=3),
    "rf_leaf10": lambda: make_random_forest(min_samples_leaf=10),
}


def plot_confusion(cm: np.ndarray, title: str, path) -> None:
    cm = np.asarray(cm)
    # Row-normalise so each row reads "of the true X recordings, what % went where".
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            color = "white" if norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(4), CLASS_ORDER)
    ax.set_yticks(range(4), CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    X, y = df[FEATURE_COLUMNS], df["label"]

    # Two-step split: carve off test first, then split the rest into train/val.
    X_rest, X_test, y_rest, y_test, rec_rest, rec_test = train_test_split(
        X, y, df["record"], test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.15 / 0.85, stratify=y_rest, random_state=RANDOM_STATE
    )
    print(f"train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    # --- Model selection on validation only ---
    val_rows = []
    for name, make in CANDIDATES.items():
        model = make().fit(X_train, y_train)
        scores = evaluate(y_val, model.predict(X_val))
        val_rows.append({"model": name, "macro_f1": scores["macro_f1"], **scores["per_class_f1"]})
        print(f"{name:14s} val macro F1 = {scores['macro_f1']:.3f}  {scores['per_class_f1']}")
    val_df = pd.DataFrame(val_rows).sort_values("macro_f1", ascending=False)
    val_df.to_csv(REPORTS_DIR / "validation_scores.csv", index=False)

    best_lr = val_df[val_df["model"].str.startswith("logreg")].iloc[0]["model"]
    best_rf = val_df[val_df["model"].str.startswith("rf")].iloc[0]["model"]
    best = val_df.iloc[0]["model"]
    print(f"Best on validation: {best}")

    # Permutation importance on validation: how much macro F1 drops when one
    # feature is shuffled. More honest than impurity importance, which favours
    # continuous features with many split points.
    best_on_train = CANDIDATES[best]().fit(X_train, y_train)
    perm = permutation_importance(
        best_on_train, X_val, y_val, scoring="f1_macro", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)

    # --- Final evaluation: refit on train+val, test ONCE ---
    # Both best LR and best RF go to test so the README can compare them, but
    # the choice of which one ships was already made on validation above.
    results = {"split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)}}
    final_models = {}
    for name in [best_lr, best_rf]:
        model = CANDIDATES[name]().fit(X_rest, y_rest)
        final_models[name] = model
        results[name] = evaluate(y_test, model.predict(X_test))
        plot_confusion(
            results[name]["confusion_matrix"],
            f"{name} on held-out test set (n={len(X_test)})",
            FIGURES_DIR / f"confusion_matrix_{name.split('_')[0]}.png",
        )
    results["selected_model"] = best
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "split_sizes"}, indent=2))

    # --- Error analysis table for the shipped model ---
    final = final_models[best]
    pred = final.predict(X_test)
    proba = final.predict_proba(X_test)
    errors = X_test.copy()
    errors.insert(0, "record", rec_test.values)
    errors.insert(1, "true", y_test.values)
    errors.insert(2, "pred", pred)
    errors.insert(3, "pred_score", proba.max(axis=1).round(3))
    errors = errors[errors["true"] != errors["pred"]]
    errors.to_csv(REPORTS_DIR / "test_errors.csv", index=False)
    print("Most common test errors (true -> pred):")
    print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8))

    fig, ax = plt.subplots(figsize=(7, 5))
    importance[::-1].plot.barh(ax=ax, color="tab:blue")
    ax.set_xlabel("Drop in validation macro F1 when shuffled")
    ax.set_title(f"Permutation importance ({best})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=130)
    plt.close(fig)

    # Everything the demo needs in one file. Class medians come from the
    # training+validation data so the app can show "typical Normal vs AF" values.
    train_df = df.loc[X_rest.index]
    artifact = {
        "model": final,
        "model_name": best,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(final.classes_),
        "class_medians": train_df.groupby("label")[FEATURE_COLUMNS].median(),
        "importance": importance,
        "test_macro_f1": results[best]["macro_f1"],
    }
    joblib.dump(artifact, MODELS_DIR / "model.joblib", compress=3)
    print("Saved", MODELS_DIR / "model.joblib")


if __name__ == "__main__":
    main()
