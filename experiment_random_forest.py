
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

OUT = Path("results")
OUT.mkdir(exist_ok=True)

TARGETS = [
    "open_pr", "open_issue", "issue_comment",
    "pr_review", "pr_review_comment", "push"
]

EXPERIMENTS = {
    "A_Baseline": (False, False),
    "B_Relational": (True, False),
    "C_Temporal": (False, True),
    "D_Full": (True, True)
}


def read_csv(path):
    return pd.read_csv(
        path, dtype={"UserID": "string"}
    )


def load_data(split, use_rel, use_temp):
    features = read_csv(
        f"data/features/{split}_features.csv"
    )

    if use_rel:
        rel = read_csv(
            f"data/features/{split}_relational.csv"
        )
        features = features.merge(
            rel, on="UserID",
            validate="one_to_one"
        )

    if use_temp:
        temp = read_csv(
            f"data/features/{split}_temporal.csv"
        )
        features = features.merge(
            temp, on="UserID",
            validate="one_to_one"
        )

    labels = read_csv(
        f"data/labels/{split}_labels.csv"
    )

    data = features.merge(
        labels, on="UserID",
        validate="one_to_one"
    )

    assert len(data) == 4465

    cols = [c for c in features if c != "UserID"]

    X = data[cols].fillna(0).astype(float)
    y = data[TARGETS].astype(int)

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

    print(f"\n{'=' * 55}")
    print(f"Experiment: {name}")
    print(f"Features: {X_train.shape[1]}")

    preds = np.zeros(
        (len(X_val), len(TARGETS)),
        dtype=int
    )

    for i, target in enumerate(TARGETS):

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train[target]
        )

        prob = model.predict_proba(X_val)[:, 1]

        pred = (prob >= 0.5).astype(int)
        preds[:, i] = pred

        p = precision_score(
            y_val[target], pred, zero_division=0
        )
        r = recall_score(
            y_val[target], pred, zero_division=0
        )
        f1 = f1_score(
            y_val[target], pred, zero_division=0
        )

        details.append({
            "Experiment": name,
            "Activity": target,
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "Predicted_Positive": int(pred.sum())
        })

        print(
            f"{target:20s} "
            f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
        )

        if name == "D_Full":
            importance = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance", ascending=False
            )

            importance.to_csv(
                OUT / f"rf_importance_{target}.csv",
                index=False
            )

    actual = y_val.to_numpy()

    macro = f1_score(
        actual, preds,
        average="macro",
        zero_division=0
    )

    micro = f1_score(
        actual, preds,
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


pd.DataFrame(details).to_csv(
    OUT / "rf_ablation_details.csv",
    index=False
)

summary = pd.DataFrame(summaries)

summary.to_csv(
    OUT / "rf_ablation_summary.csv",
    index=False
)

print(f"\n{'=' * 55}")
print("RANDOM FOREST ABLATION SUMMARY")
print(summary.to_string(index=False))

print("\nExperiment completed.")
