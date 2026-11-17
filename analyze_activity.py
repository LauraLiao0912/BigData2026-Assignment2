
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/db")

# 每張活動表使用的觀察時間欄位
TABLES = {
    "PullRequests": "first_seen_at",
    "Issues": "first_seen_at",
    "IssueComments": "created_at",
    "PRReviews": "first_seen_at",
    "PRReviewComments": "created_at",
    "Pushes": "created_at",
}

# 台灣時區：UTC+8
for table_name, time_col in TABLES.items():

    df = pd.read_parquet(
        DATA_DIR / f"{table_name}.parquet",
        columns=[time_col]
    )

    timestamps = pd.to_datetime(
        df[time_col], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Taipei")

    daily_counts = (
        timestamps.dt.date
        .value_counts()
        .sort_index()
    )

    print(f"\n{'=' * 50}")
    print(f"Table: {table_name}")
    print(f"Total rows: {len(df):,}")
    print(f"Time column: {time_col}")
    print("\nDaily Activity Counts:")
    print(daily_counts.to_string())

print("\nAnalysis completed.")
