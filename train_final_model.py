
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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

FEATURE_GROUPS = [
    ("features", "features"),
    ("temporal", "temporal"),
]


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={"UserID": "string"}
    )


def load_features(split):

    features = read_csv(
        DATA / "features" / f"{split}_features.csv"
    )

    temporal = read_csv(
        DATA / "features" / f"{split}_temporal.csv"
    )

    assert features["UserID"].is_unique
    assert temporal["UserID"].is_unique

    merged = features.merge(
        temporal,
        on="UserID",
        validate="one_to_one"
    )

    assert len(merged) == len(features)

    return merged


def load_training_split(split):

    X = load_features(split)

    y = read_csv(
        DATA / "labels" / f"{split}_labels.csv"
    )

    assert y["UserID"].is_unique

    merged = X.merge(
        y,
        on="UserID",
        validate="one_to_one"
    )

    assert len(merged) == len(X) == len(y)

    return merged


# 讀取兩個歷史訓練週
train = load_training_split("train")
validation = load_training_split("validation")

# 合併歷史樣本重新訓練
training_data = pd.concat(
    [train, validation],
    ignore_index=True
)

# 正式預測特徵
final = load_features("final")

feature_columns = [
    c for c in final.columns
    if c != "UserID"
]

assert len(feature_columns) == 40

X_train = training_data[feature_columns].astype(float)
X_final = final[feature_columns].astype(float)

assert X_train.notna().all().all()
assert X_final.notna().all().all()
assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_final.to_numpy()).all()

print("Training shape:", X_train.shape)
print("Prediction shape:", X_final.shape)

# 從官方測試名單取得使用者順序
test = read_csv(
    "submission/test_input.csv"
)

submission = test[["UserID"]].copy()

# 確保預測資料與測試使用者完全一致
final_predictions = test[["UserID"]].merge(
    final[["UserID"]],
    on="UserID",
    validate="one_to_one"
)

assert len(final_predictions) == len(test)

for target in TARGETS:

    print(f"\nTraining model: {target}")

    y_train = training_data[target].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_final)[:, 1]

    prediction_map = pd.Series(
        (probabilities >= 0.5).astype(int),
        index=final["UserID"].astype(str)
    )

    submission[target] = (
        submission["UserID"]
        .astype(str)
        .map(prediction_map)
    )

    assert submission[target].notna().all()

    submission[target] = (
        submission[target].astype(int)
    )

    joblib.dump(
        model,
        OUT / f"final_rf_{target}.joblib"
    )

    print(
        f"Predicted positives: "
        f"{submission[target].sum()}"
    )


# 檢查輸出格式
assert len(submission) == len(test)
assert submission["UserID"].tolist() == test["UserID"].tolist()
assert list(submission.columns) == ["UserID"] + TARGETS

for target in TARGETS:
    assert submission[target].isin([0, 1]).all()

# 先建立暫定檔案，確認流程正常
output_path = OUT / "output_draft.csv"

submission.to_csv(
    output_path,
    index=False
)

print(f"\nSaved: {output_path}")
print("Rows:", len(submission))
print("Columns:", len(submission.columns))
print("\nFinal prediction generation completed.")
