
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path("data/db")
OUT = Path("data/multiweek")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "open_pr", "open_issue", "issue_comment",
    "pr_review", "pr_review_comment", "push"
]

CONFIG = {
    "open_pr": (
        "PullRequests", "author_id",
        "created_at", "first_seen_at"
    ),
    "open_issue": (
        "Issues", "author_id",
        "created_at", "first_seen_at"
    ),
    "issue_comment": (
        "IssueComments", "author_id",
        "created_at", "created_at"
    ),
    "pr_review": (
        "PRReviews", "reviewer_id",
        "submitted_at", "first_seen_at"
    ),
    "pr_review_comment": (
        "PRReviewComments", "author_id",
        "created_at", "created_at"
    ),
    "push": (
        "Pushes", "user_id",
        "created_at", "created_at"
    ),
}


def to_utc(date):
    return pd.Timestamp(
        date, tz="Asia/Taipei"
    ).tz_convert("UTC")


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

for target, config in CONFIG.items():
    table, user_col, event_col, seen_col = config

    df = pd.read_parquet(DB / f"{table}.parquet")

    if target == "open_issue":
        df = df[
            df["is_pull_request"].eq(False).fillna(False)
        ].copy()

    df["UserID"] = df[user_col].map(user_map)

    df["event_time"] = pd.to_datetime(
        df[event_col], utc=True, errors="coerce"
    )

    df["seen_time"] = pd.to_datetime(
        df[seen_col], utc=True, errors="coerce"
    )

    events[target] = df[
        ["UserID", "repo_id", "event_time", "seen_time"]
    ].dropna(subset=["UserID", "seen_time"])

    print(target, len(events[target]))


def build_week(cutoff_date, with_labels=True):

    cutoff = to_utc(cutoff_date)
    start_7d = cutoff - pd.Timedelta(days=7)
    start_30d = cutoff - pd.Timedelta(days=30)

    result = test[["UserID"]].copy()
    result["cutoff"] = cutoff_date

    relational_parts = []

    for target, df in events.items():

        # Only records observable before prediction time
        history = df[
            (df["seen_time"] >= start_30d) &
            (df["seen_time"] < cutoff)
        ].copy()

        for days, start in [
            (7, start_7d),
            (30, start_30d)
        ]:
            window = history[
                history["seen_time"] >= start
            ]

            counts = window.groupby("UserID").size()

            result[f"{target}_count_{days}d"] = (
                result["UserID"].map(counts)
                .fillna(0).astype(int)
            )

        # Recency within 30 days
        last_seen = history.groupby("UserID")[
            "seen_time"
        ].max()

        recency = (
            (cutoff - last_seen).dt.total_seconds()
            / 86400
        )

        result[f"{target}_recency_30d"] = (
            result["UserID"].map(recency)
            .fillna(31).clip(0, 31)
        )

        # Active days within 30 days
        history["local_date"] = (
            history["seen_time"]
            .dt.tz_convert("Asia/Taipei")
            .dt.date
        )

        active_days = history.groupby("UserID")[
            "local_date"
        ].nunique()

        result[f"{target}_active_days_30d"] = (
            result["UserID"].map(active_days)
            .fillna(0).astype(int)
        )

        relational_parts.append(
            history[["UserID", "repo_id"]]
        )

        # Labels: events observed within target week
        if with_labels:
            end = cutoff + pd.Timedelta(days=7)

            positive = df[
                (df["event_time"] >= cutoff) &
                (df["event_time"] < end) &
                (df["seen_time"] >= cutoff) &
                (df["seen_time"] < end)
            ]["UserID"].dropna().unique()

            result[target] = (
                result["UserID"].isin(positive)
                .astype(int)
            )

    # Relational features
    pairs = pd.concat(
        relational_parts, ignore_index=True
    ).dropna(subset=["repo_id"]).drop_duplicates()

    repo_users = pairs.groupby("repo_id")[
        "UserID"
    ].nunique()

    pairs["peer_count"] = (
        pairs["repo_id"].map(repo_users) - 1
    )

    unique_repos = pairs.groupby("UserID")[
        "repo_id"
    ].nunique()

    max_peers = pairs.groupby("UserID")[
        "peer_count"
    ].max()

    result["unique_repos_30d"] = (
        result["UserID"].map(unique_repos)
        .fillna(0).astype(int)
    )

    result["max_repo_peers_30d"] = (
        result["UserID"].map(max_peers)
        .fillna(0).astype(int)
    )

    return result


TRAIN_CUTOFFS = [
    "2026-09-20",
    "2026-09-27",
    "2026-10-04",
    "2026-10-11",
    "2026-10-18",
    "2026-10-25",
    "2026-11-01",
]

train_parts = []

for date in TRAIN_CUTOFFS:
    print("Building train week:", date)
    train_parts.append(build_week(date))

train = pd.concat(train_parts, ignore_index=True)

validation = build_week("2026-11-08")

# Draft prediction uses data through Nov 16, 00:00
final = build_week(
    "2026-11-16", with_labels=False
)

train.to_csv(OUT / "train.csv", index=False)
validation.to_csv(OUT / "validation.csv", index=False)
final.to_csv(OUT / "final.csv", index=False)

print("\n=== Dataset Summary ===")

for name, df in [
    ("Train", train),
    ("Validation", validation),
    ("Final", final)
]:
    print(name, df.shape)
    print("Missing values:", df.isna().sum().sum())

print("\nTraining label distribution:")
print(train[TARGETS].sum().to_string())

print("\nValidation label distribution:")
print(validation[TARGETS].sum().to_string())

print("\nCompleted.")
