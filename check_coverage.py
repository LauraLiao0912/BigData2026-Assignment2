
import pandas as pd

# Official test users
test = pd.read_csv(
    "submission/test_input.csv",
    dtype={"UserID": "string"}
)

# Newly preprocessed users
users = pd.read_parquet(
    "data/db/Users.parquet"
)

test_ids = set(test["UserID"].dropna().astype(str))
observed_ids = set(
    users["github_user_id"].dropna().astype(str)
)

matched = test_ids & observed_ids
missing = test_ids - observed_ids

print("=== Three-Month Dataset Coverage ===")
print(f"Test users: {len(test_ids):,}")
print(f"Database users: {len(users):,}")
print(f"Matched test users: {len(matched):,}")
print(f"Missing test users: {len(missing):,}")

coverage = len(matched) / len(test_ids) * 100
print(f"Coverage: {coverage:.2f}%")

print("\n=== Comparison ===")
print("Previous 22-day coverage: 48.40%")
print(f"New 92-day coverage: {coverage:.2f}%")
print(f"Improvement: {coverage - 48.40:+.2f} percentage points")
