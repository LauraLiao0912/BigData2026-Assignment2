
import gzip
import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path("raw_data")

# 比較有 Pushes 紀錄與沒有 Pushes 紀錄的日期
# 以下檔名使用 UTC
FILES = [
    "2026-10-25-0.json.gz",
    "2026-10-28-0.json.gz",
    "2026-11-10-0.json.gz",
    "2026-11-15-0.json.gz",
]

for filename in FILES:
    path = RAW_DIR / filename

    if not path.exists():
        print(f"File not found: {filename}")
        continue

    event_counts = Counter()

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            event_type = event.get("type", "Unknown")
            event_counts[event_type] += 1

    print(f"\n{'=' * 45}")
    print(f"File: {filename}")
    print(f"Total events: {sum(event_counts.values()):,}")
    print(f"PushEvent: {event_counts['PushEvent']:,}")

    print("\nTop 10 event types:")
    for name, count in event_counts.most_common(10):
        print(f"{name:30s} {count:,}")

print("\nAnalysis completed.")
