
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
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


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={"UserID": "string"}
    )


def load_dataset(split):
    features = read_csv(
        DATA / "features" / f"{split}_features.csv"
    )

    labels = read_csv(
        DATA / "labels" / f"{split}_labels.csv"
    )

    assert features["UserID"].is_unique
    assert labels["UserID"].is_unique

    merged = features.merge(
        labels,
        on="UserID",
        how="inner",
        validate="one_to_one"
    )

    assert len(merged) == len(features) == len(labels)

    feature_cols = [
        c for c in features.columns
        if c != "UserID"
    ]

    X = merged[feature_cols].astype(float)
    y = merged[TARGETS].astype(int)

    return X, y, merged["UserID"]


X_train, y_train, train_ids = load_dataset("train")
X_val, y_val, val_ids = load_dataset("validation")

assert list(X_train.columns) == list(X_val.columns)

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)
print("Validation X:", X_val.shape)
print("Validation y:", y_val.shape)

results = []
predictions = pd.DataFrame({"UserID": val_ids})

for target in TARGETS:

    print(f"\n{'=' * 55}")
    print(f"Activity: {target}")

    y_tr = y_train[target].to_numpy()
    y_va = y_val[target].to_numpy()

    # Model A: Majority Baseline
    majority_pred = np.zeros(len(y_va), dtype=int)

    # Model B: Logistic Regression
    # 紀錄次數先做 log1p，減少極端活躍使用者的影響
    X_tr = np.log1p(X_train)
    X_va = np.log1p(X_val)

    if len(np.unique(y_tr)) < 2:
        print("Only one training class; using constant prediction.")
        logistic_pred = np.full(
            len(y_va), y_tr[0], dtype=int
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )

        model.fit(X_tr, y_tr)

        probabilities = model.predict_proba(X_va)[:, 1]

        # 第一版固定閾值，不使用驗證集調整
        logistic_pred = (
            probabilities >= 0.5
        ).astype(int)

        joblib.dump(
            model,
            OUT / f"baseline_{target}.joblib"
        )

    predictions[target] = logistic_pred

    for name, pred in [
        ("Majority", majority_pred),
        ("LogisticRegression", logistic_pred),
    ]:

        precision = precision_score(
            y_va, pred, zero_division=0
        )
        recall = recall_score(
            y_va, pred, zero_division=0
        )
        f1 = f1_score(
            y_va, pred, zero_division=0
        )

        results.append({
            "Activity": target,
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Actual_Positive": int(y_va.sum()),
            "Predicted_Positive": int(pred.sum()),
        })

        print(
            f"{name:20s} "
            f"Precision={precision:.4f} "
            f"Recall={recall:.4f} "
            f"F1={f1:.4f} "
            f"Predicted Positive={pred.sum()}"
        )


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "baseline_metrics.csv",
    index=False
)

predictions.to_csv(
    OUT / "baseline_validation_predictions.csv",
    index=False
)

print(f"\n{'=' * 55}")
print("Overall Validation Results")

for model_name in ["Majority", "LogisticRegression"]:
    pred = np.zeros_like(y_val.to_numpy())

    if model_name == "LogisticRegression":
        pred = predictions[TARGETS].to_numpy()

    macro_f1 = f1_score(
        y_val.to_numpy(),
        pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_val.to_numpy(),
        pred,
        average="micro",
        zero_division=0
    )

    score = (macro_f1 + micro_f1) / 2

    print(f"\nModel: {model_name}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Reference Score: {score:.4f}")

print("\nTraining completed.")
