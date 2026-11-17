
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push"
]


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={"UserID": "string"}
    )


def load_dataset(split, use_relational=False):

    features = read_csv(
        f"data/features/{split}_features.csv"
    )

    labels = read_csv(
        f"data/labels/{split}_labels.csv"
    )

    if use_relational:
        relational = read_csv(
            f"data/features/{split}_relational.csv"
        )

        features = features.merge(
            relational,
            on="UserID",
            how="left",
            validate="one_to_one"
        )

    data = features.merge(
        labels,
        on="UserID",
        how="inner",
        validate="one_to_one"
    )

    assert len(data) == len(labels) == 4465

    feature_columns = [
        c for c in features.columns
        if c != "UserID"
    ]

    X = np.log1p(
        data[feature_columns].fillna(0).astype(float)
    )

    y = data[TARGETS].astype(int)

    return X, y


def run_experiment(name, use_relational):

    X_train, y_train = load_dataset(
        "train", use_relational
    )

    X_val, y_val = load_dataset(
        "validation", use_relational
    )

    assert list(X_train.columns) == list(X_val.columns)

    print(f"\n{'=' * 55}")
    print(f"Model: {name}")
    print(f"Features: {X_train.shape[1]}")

    predictions = np.zeros(
        (len(X_val), len(TARGETS)),
        dtype=int
    )

    results = []

    for i, target in enumerate(TARGETS):

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )

        model.fit(
            X_train,
            y_train[target]
        )

        prob = model.predict_proba(X_val)[:, 1]

        pred = (prob >= 0.5).astype(int)

        predictions[:, i] = pred

        p = precision_score(
            y_val[target], pred, zero_division=0
        )

        r = recall_score(
            y_val[target], pred, zero_division=0
        )

        f1 = f1_score(
            y_val[target], pred, zero_division=0
        )

        results.append({
            "Model": name,
            "Activity": target,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "Predicted_Positive": int(pred.sum())
        })

        print(
            f"{target:20s} "
            f"P={p:.4f} "
            f"R={r:.4f} "
            f"F1={f1:.4f}"
        )

    actual = y_val.to_numpy()

    macro = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    micro = f1_score(
        actual,
        predictions,
        average="micro",
        zero_division=0
    )

    score = (macro + micro) / 2

    print(f"\nMacro-F1: {macro:.4f}")
    print(f"Micro-F1: {micro:.4f}")
    print(f"Reference Score: {score:.4f}")

    return results, {
        "Model": name,
        "Num_Features": X_train.shape[1],
        "Macro_F1": macro,
        "Micro_F1": micro,
        "Reference_Score": score
    }


# Experiment A: Original Baseline
baseline_results, baseline_summary = run_experiment(
    "Baseline",
    use_relational=False
)

# Experiment B: Add Relational Features
relational_results, relational_summary = run_experiment(
    "Baseline + Relational",
    use_relational=True
)

# 儲存每項活動的結果
detail = pd.DataFrame(
    baseline_results + relational_results
)

detail.to_csv(
    OUT / "relational_ablation_details.csv",
    index=False
)

# 儲存整體比較結果
summary = pd.DataFrame([
    baseline_summary,
    relational_summary
])

summary.to_csv(
    OUT / "relational_ablation_summary.csv",
    index=False
)

print(f"\n{'=' * 55}")
print("Ablation Study Summary")
print(summary.to_string(index=False))

improvement = (
    relational_summary["Reference_Score"]
    - baseline_summary["Reference_Score"]
)

print(f"\nReference Score Difference: {improvement:+.4f}")
print("\nExperiment completed.")
