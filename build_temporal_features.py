
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/db")
OUT = Path("data/features")
OUT.mkdir(parents=True, exist_ok=True)

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

test = pd.read_csv(
    "submission/test_input.csv",
    dtype={"UserID": "string"}
)

users = pd.read_parquet(
    DB / "Users.parquet"
)

user_map = (
    users[users["is_cohort"].fillna(False)]
    .set_index("user_id")["github_user_id"]
    .astype("string")
)

events = {}

for activity, (table, user_col, time_col) in CONFIG.items():
    df = pd.read_parquet(DB / f"{table}.parquet")

    if activity == "open_issue":
        df = df[
            df["is_pull_request"].eq(False).fillna(False)
        ].copy()

    df["UserID"] = df[user_col].map(user_map)

    df["time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce"
    )

    events[activity] = df[
        ["UserID", "time"]
    ].dropna()

    print(f"Loaded {activity}: {len(events[activity])} rows")


def build_temporal(cutoff_date, filename):

    cutoff = pd.Timestamp(
        cutoff_date,
        tz="Asia/Taipei"
    ).tz_convert("UTC")

    start_7d = cutoff - pd.Timedelta(days=7)
    start_3d = cutoff - pd.Timedelta(days=3)

    features = test[["UserID"]].copy()

    for activity, df in events.items():

        # 僅使用預測時間之前的七天紀錄
        history = df[
            (df["time"] >= start_7d) &
            (df["time"] < cutoff)
        ].copy()

        # 1. Recency
        last_time = history.groupby("UserID")["time"].max()

        days_since = (
            (cutoff - last_time).dt.total_seconds()
            / 86400
        )

        # 8 代表七天內沒有觀察到活動
        features[f"{activity}_recency_7d"] = (
            features["UserID"].map(days_since)
            .fillna(8.0)
            .clip(lower=0, upper=8)
        )

        # 2. Active Days
        # 使用台灣時間判定一天的日期
        history["local_date"] = (
            history["time"]
            .dt.tz_convert("Asia/Taipei")
            .dt.date
        )

        active_days = (
            history.groupby("UserID")["local_date"]
            .nunique()
        )

        features[f"{activity}_active_days_7d"] = (
            features["UserID"]
            .map(active_days)
            .fillna(0)
            .astype(int)
        )

        # 3. Activity Trend
        recent = history[
            history["time"] >= start_3d
        ]

        previous = history[
            history["time"] < start_3d
        ]

        recent_counts = recent.groupby("UserID").size()
        previous_counts = previous.groupby("UserID").size()

        recent_values = (
            features["UserID"]
            .map(recent_counts)
            .fillna(0)
        )

        previous_values = (
            features["UserID"]
            .map(previous_counts)
            .fillna(0)
        )

        # 加 1 平滑，避免分母為 0
        trend = (
            (recent_values + 1) /
            (previous_values + 1)
        )

        features[f"{activity}_trend_3v4d"] = trend

    output_path = OUT / filename
    features.to_csv(output_path, index=False)

    print(f"\nCutoff: {cutoff_date}")
    print(f"Saved: {output_path}")
    print(f"Shape: {features.shape}")
    print(
        f"Missing values: "
        f"{features.isna().sum().sum()}"
    )

    print("\nFeature Statistics:")
    print(
        features.drop(columns="UserID")
        .describe()
        .round(3)
        .to_string()
    )

    return features


train = build_temporal(
    "2026-11-01",
    "train_temporal.csv"
)

validation = build_temporal(
    "2026-11-08",
    "validation_temporal.csv"
)

final = build_temporal(
    "2026-11-16",
    "final_temporal.csv"
)

print("\nTemporal feature generation completed.")
