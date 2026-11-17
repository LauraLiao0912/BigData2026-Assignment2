
from pathlib import Path
import pandas as pd

DB = Path("data/db")
OUT = Path("data/features")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "PullRequests": ("author_id", "first_seen_at"),
    "Issues": ("author_id", "first_seen_at"),
    "IssueComments": ("author_id", "created_at"),
    "PRReviews": ("reviewer_id", "first_seen_at"),
    "PRReviewComments": ("author_id", "created_at"),
    "Pushes": ("user_id", "created_at"),
}

users = pd.read_parquet(
    DB / "Users.parquet",
    columns=["user_id", "github_user_id", "is_cohort"]
)

user_map = (
    users[users["is_cohort"].fillna(False)]
    .set_index("user_id")["github_user_id"]
    .astype("string")
)

events = []

for table, (user_col, time_col) in CONFIG.items():
    df = pd.read_parquet(
        DB / f"{table}.parquet"
    )

    if table == "Issues":
        df = df[
            df["is_pull_request"].eq(False).fillna(False)
        ].copy()

    df["UserID"] = df[user_col].map(user_map)
    df["time"] = pd.to_datetime(
        df[time_col], utc=True, errors="coerce"
    )

    df = df[
        ["UserID", "repo_id", "time"]
    ].dropna()

    events.append(df)

all_events = pd.concat(events, ignore_index=True)

test = pd.read_csv(
    "submission/test_input.csv",
    dtype={"UserID": "string"}
)


def build(cutoff_date, output_name):
    cutoff = pd.Timestamp(
        cutoff_date, tz="Asia/Taipei"
    ).tz_convert("UTC")

    start = cutoff - pd.Timedelta(days=7)

    recent = all_events[
        (all_events["time"] >= start) &
        (all_events["time"] < cutoff)
    ].copy()

    features = test[["UserID"]].copy()

    # User -> Repository 關係
    pairs = recent[
        ["UserID", "repo_id"]
    ].drop_duplicates()

    # 每位使用者參與多少不同 Repository
    unique_repos = pairs.groupby("UserID")[
        "repo_id"
    ].nunique()

    # 每個 Repository 的活動次數
    repo_events = recent.groupby("repo_id").size()

    # 每個 Repository 的不同活躍使用者數
    repo_users = pairs.groupby("repo_id")[
        "UserID"
    ].nunique()

    pairs = pairs.copy()

    # 一個 Repository 中，除了自己還有多少人
    pairs["peer_count"] = (
        pairs["repo_id"].map(repo_users) - 1
    )

    pairs["repo_events"] = (
        pairs["repo_id"].map(repo_events)
    )

    peer_max = pairs.groupby("UserID")[
        "peer_count"
    ].max()

    peer_mean = pairs.groupby("UserID")[
        "peer_count"
    ].mean()

    repo_total = pairs.groupby("UserID")[
        "repo_events"
    ].sum()

    feature_maps = {
        "unique_repos_7d": unique_repos,
        "repo_peer_max_7d": peer_max,
        "repo_peer_mean_7d": peer_mean,
        "repo_total_events_7d": repo_total,
    }

    for name, mapping in feature_maps.items():
        features[name] = (
            features["UserID"]
            .map(mapping)
            .fillna(0)
        )

    output_path = OUT / output_name
    features.to_csv(output_path, index=False)

    print(f"\nCutoff: {cutoff_date}")
    print("Shape:", features.shape)
    print("Missing values:", features.isna().sum().sum())

    print("\nFeature statistics:")
    print(
        features.drop(columns="UserID")
        .describe()
        .round(2)
        .to_string()
    )


build(
    "2026-11-01",
    "train_relational.csv"
)

build(
    "2026-11-08",
    "validation_relational.csv"
)

print("\nRelational feature generation completed.")
