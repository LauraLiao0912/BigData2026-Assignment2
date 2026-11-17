
from pathlib import Path
import pandas as pd

DB = Path("data/db")
OUT = Path("data/features")
OUT.mkdir(parents=True, exist_ok=True)

# 第一版使用 1、3、7 天的歷史特徵
# 14 天視窗留到後續實驗，避免第一週資料不足
WINDOWS = [1, 3, 7]

# 活動資料表、使用者欄位、觀察時間
CONFIG = {
    "open_pr": (
        "PullRequests", "author_id", "first_seen_at"
    ),
    "open_issue": (
        "Issues", "author_id", "first_seen_at"
    ),
    "issue_comment": (
        "IssueComments", "author_id", "created_at"
    ),
    "pr_review": (
        "PRReviews", "reviewer_id", "first_seen_at"
    ),
    "pr_review_comment": (
        "PRReviewComments", "author_id", "created_at"
    ),
    "push": (
        "Pushes", "user_id", "created_at"
    ),
}


def taiwan_to_utc(date_str):
    return pd.Timestamp(
        date_str, tz="Asia/Taipei"
    ).tz_convert("UTC")


# 讀取官方測試使用者
test = pd.read_csv(
    "submission/test_input.csv",
    dtype={"UserID": "string"}
)

# 讀取使用者對照表
users = pd.read_parquet(DB / "Users.parquet")

users = users[
    users["is_cohort"].fillna(False)
].copy()

user_map = (
    users.set_index("user_id")["github_user_id"]
    .astype("string")
)

# 預先讀取活動表
events = {}

for activity, (table, user_col, time_col) in CONFIG.items():
    df = pd.read_parquet(DB / f"{table}.parquet")

    # Issue 不可以包含 PR
    if activity == "open_issue":
        df = df[
            df["is_pull_request"].eq(False).fillna(False)
        ].copy()

    # 轉換成官方 UserID
    df["UserID"] = df[user_col].map(user_map)

    # 原始 Parquet 時間視為 UTC
    df["event_time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["UserID", "event_time"]
    )

    events[activity] = df[["UserID", "event_time"]]

    print(
        f"Loaded {activity:20s}: {len(df):,} rows"
    )


def build_features(prediction_date, filename):
    cutoff = taiwan_to_utc(prediction_date)

    # 從所有測試使用者開始
    features = test[["UserID"]].copy()

    print(f"\nPrediction cutoff: {prediction_date}")

    for activity, df in events.items():

        # 嚴格排除預測時間之後的資料
        history = df[df["event_time"] < cutoff]

        for days in WINDOWS:
            start = cutoff - pd.Timedelta(days=days)

            recent = history[
                history["event_time"] >= start
            ]

            counts = (
                recent.groupby("UserID")
                .size()
            )

            column = f"{activity}_count_{days}d"

            features[column] = (
                features["UserID"]
                .map(counts)
                .fillna(0)
                .astype(int)
            )

    # 加入跨活動總次數
    for days in WINDOWS:
        cols = [
            f"{activity}_count_{days}d"
            for activity in CONFIG
        ]

        features[f"total_count_{days}d"] = (
            features[cols].sum(axis=1)
        )

    # 是否有最近七天活動
    features["active_last_7d"] = (
        features["total_count_7d"] > 0
    ).astype(int)

    output_path = OUT / filename
    features.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {features.shape}")
    print(
        "Users active in last 7 days:",
        features["active_last_7d"].sum()
    )

    return features


# Train：11/01 開始預測
train = build_features(
    "2026-11-01",
    "train_features.csv"
)

# Validation：11/08 開始預測
validation = build_features(
    "2026-11-08",
    "validation_features.csv"
)

final = build_features(
    "2026-11-16",
    "final_features.csv"
)

print("\nFeature generation completed.")
