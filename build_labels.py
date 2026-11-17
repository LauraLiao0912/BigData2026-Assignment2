
from pathlib import Path
import pandas as pd

# 資料路徑
DB = Path("data/db")
OUT = Path("data/labels")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]

# 欄位：資料表、使用者、事件時間、可觀察時間
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


def to_utc_naive(date_str):
    """將台灣時間轉為無時區的 UTC Timestamp。"""
    return (
        pd.Timestamp(date_str, tz="Asia/Taipei")
        .tz_convert("UTC")
        .tz_localize(None)
    )


def build_labels(start_date, end_date, filename):
    start = to_utc_naive(start_date)
    end = to_utc_naive(end_date)

    # 以官方測試使用者為固定分析對象
    test = pd.read_csv(
        "submission/test_input.csv",
        dtype={"UserID": "string"}
    )

    users = pd.read_parquet(
        DB / "Users.parquet",
        columns=["user_id", "github_user_id", "is_cohort"]
    )

    users = users[users["is_cohort"].fillna(False)]

    # 將資料庫內部 user_id 轉回官方 UserID
    user_map = users.set_index("user_id")[
        "github_user_id"
    ].astype("string")

    labels = test[["UserID"]].copy()

    for target in TARGETS:
        labels[target] = 0

    print(f"\n{'=' * 50}")
    print(f"Period: {start_date} -> {end_date}")

    for target, config in CONFIG.items():
        table, user_col, event_col, seen_col = config

        df = pd.read_parquet(DB / f"{table}.parquet")

        event_time = pd.to_datetime(
            df[event_col], errors="coerce"
        )
        seen_time = pd.to_datetime(
            df[seen_col], errors="coerce"
        )

        # 事件確實發生在目標週
        event_mask = (
            (event_time >= start) &
            (event_time < end)
        )

        # 事件也必須在該週被觀察到
        seen_mask = (
            (seen_time >= start) &
            (seen_time < end)
        )

        mask = event_mask & seen_mask

        # Issue 不能是 Pull Request
        if target == "open_issue":
            mask &= df["is_pull_request"].eq(False).fillna(False)

        selected = df.loc[mask, [user_col]].copy()

        selected["UserID"] = (
            selected[user_col].map(user_map)
        )

        positive_users = set(
            selected["UserID"].dropna().astype(str)
        )

        labels[target] = (
            labels["UserID"].isin(positive_users)
            .astype(int)
        )

        print(
            f"{target:20s} "
            f"Positive users: {labels[target].sum():5d}"
        )

    output_path = OUT / filename

    labels.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Users: {len(labels):,}")

    return labels


# 訓練週
train = build_labels(
    "2026-11-01",
    "2026-11-08",
    "train_labels.csv"
)

# 驗證週
validation = build_labels(
    "2026-11-08",
    "2026-11-15",
    "validation_labels.csv"
)

print("\nLabel generation completed.")
