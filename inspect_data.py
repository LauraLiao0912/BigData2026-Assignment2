
from pathlib import Path
import pandas as pd

# 讀取八張資料表
data_dir = Path("data/db")

tables = {}

for path in sorted(data_dir.glob("*.parquet")):
    df = pd.read_parquet(path)
    tables[path.stem] = df

    print(f"\n{'-' * 50}")
    print(f"Table: {path.stem}")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

# 檢查測試使用者的資料涵蓋率
print("\n" + "-" * 50)
print("Test User Coverage")

test = pd.read_csv("submission/test_input.csv")
users = tables["Users"]

test_ids = set(test["UserID"].astype(str))
observed_ids = set(users["github_user_id"].astype(str))

matched = test_ids & observed_ids

print(f"Test users: {len(test_ids):,}")
print(f"Users in database: {len(users):,}")
print(f"Matched test users: {len(matched):,}")
print(f"Missing test users: {len(test_ids - matched):,}")
print(f"Coverage: {len(matched) / len(test_ids):.2%}")
