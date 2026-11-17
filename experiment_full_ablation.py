
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

# Output folder
OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]

EXPERIMENTS = {
    "A_Baseline": (False, False),
    "B_Relational": (True, False),
    "C_Temporal": (False, True),
    "D_Full": (True, True),
}


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={"UserID": "string"}
    )


def load_data(split, relational, temporal):

    # Historical features
    features = read_csv(
        f"data/features/{split}_features.csv"
    )

    if relational:
        rel = read_csv(
            f"data/features/{split}_relational.csv"
        )

        features = features.merge(
            rel,
            on="UserID",
            how="left",
            validate="one_to_one"
        )

    if temporal:
        temp = read_csv(
            f"data/features/{split}_temporal.csv"
        )

        features = features.merge(
            temp,
            on="UserID",
            how="left",
            validate="one_to_one"
        )

    labels = read_csv(
        f"data/labels/{split}_labels.csv"
    )

    data = features.merge(
        labels,
        on="UserID",
        how="inner",
        validate="one_to_one"
    )

    assert len(data) == len(features) == len(labels)

    feature_cols = [
        c for c in features.columns
        if c != "UserID"
    ]

    # Apply the same transformation to all experiments
    X = np.log1p(
        data[feature_cols]
        .fillna(0)
        .astype(float)
    )

    y = data[TARGETS].astype(int)

    assert np.isfinite(X.to_numpy()).all()

    return X, y


details = []
summaries = []

for name, (use_rel, use_temp) in EXPERIMENTS.items():

    X_train, y_train = load_data(
        "train", use_rel, use_temp
    )

    X_val, y_val = load_data(
        "validation", use_rel, use_temp
    )

    assert list(X_train.columns) == list(X_val.columns)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"Number of Features: {X_train.shape[1]}")

    predictions = np.zeros(
        (len(X_val), len(TARGETS)),
        dtype=int
    )

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

        probability = model.predict_proba(
            X_val
        )[:, 1]

        pred = (probability >= 0.5).astype(int)

        predictions[:, i] = pred

        precision = precision_score(
            y_val[target],
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_val[target],
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val[target],
            pred,
            zero_division=0
        )

        details.append({
            "Experiment": name,
            "Activity": target,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Predicted_Positive": int(pred.sum())
        })

        print(
            f"{target:20s} "
            f"P={precision:.4f} "
            f"R={recall:.4f} "
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

    summaries.append({
        "Experiment": name,
        "Features": X_train.shape[1],
        "Macro_F1": macro,
        "Micro_F1": micro,
        "Reference_Score": score
    })

    print(f"\nMacro-F1: {macro:.4f}")
    print(f"Micro-F1: {micro:.4f}")
    print(f"Reference Score: {score:.4f}")


# Save results
detail_df = pd.DataFrame(details)
summary_df = pd.DataFrame(summaries)

detail_df.to_csv(
    OUT / "full_ablation_details.csv",
    index=False
)

summary_df.to_csv(
    OUT / "full_ablation_summary.csv",
    index=False
)

print(f"\n{'=' * 60}")
print("FINAL ABLATION COMPARISON")
print(summary_df.to_string(index=False))

baseline_score = summary_df.iloc[0]["Reference_Score"]

print("\nDifference from Baseline:")

for _, row in summary_df.iterrows():
    diff = row["Reference_Score"] - baseline_score

    print(
        f"{row['Experiment']:20s} "
        f"{diff:+.4f}"
    )

print("\nAll experiments completed.")
