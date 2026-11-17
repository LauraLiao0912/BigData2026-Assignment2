
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
OUT.mkdir(exist_ok=True)

TARGETS = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push"
]

THRESHOLDS = [0.3, 0.5, 0.7]

def load_split(split):
    X = pd.read_csv(
        f"data/features/{split}_features.csv",
        dtype={"UserID": "string"}
    )
    y = pd.read_csv(
        f"data/labels/{split}_labels.csv",
        dtype={"UserID": "string"}
    )

    df = X.merge(
        y, on="UserID",
        validate="one_to_one"
    )

    assert len(df) == len(X) == len(y)

    columns = [c for c in X.columns if c != "UserID"]

    return (
        np.log1p(df[columns].astype(float)),
        df[TARGETS].astype(int)
    )


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

records = []

for target in TARGETS:
    print(f"\nActivity: {target}")

    for weight_name, weight in [
        ("None", None),
        ("Balanced", "balanced")
    ]:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=weight,
                max_iter=2000,
                random_state=42
            )
        )

        model.fit(X_train, y_train[target])

        prob = model.predict_proba(X_val)[:, 1]

        for threshold in THRESHOLDS:
            pred = (prob >= threshold).astype(int)

            records.append({
                "Activity": target,
                "Class_Weight": weight_name,
                "Threshold": threshold,
                "Precision": precision_score(
                    y_val[target], pred,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_val[target], pred,
                    zero_division=0
                ),
                "F1": f1_score(
                    y_val[target], pred,
                    zero_division=0
                ),
                "Predicted_Positive": int(pred.sum())
            })


results = pd.DataFrame(records)

results.to_csv(
    OUT / "threshold_experiments.csv",
    index=False
)

print("\nAll Experimental Results:")
print(results.to_string(index=False))

print("\nBest Observed Configuration Per Activity:")
best_idx = results.groupby("Activity")["F1"].idxmax()

best = results.loc[best_idx].sort_values("Activity")

print(best.to_string(index=False))

print("\nThese are exploratory validation results.")
print("Do not treat the selected thresholds as unbiased test results.")
