
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import gzip

# Taiwan: 2026/10/25 00:00 ~ 2026/11/16 00:00
# GH Archive uses UTC.

start = datetime(2026, 10, 24, 16, tzinfo=timezone.utc)
end = datetime(2026, 11, 15, 16, tzinfo=timezone.utc)

output_dir = Path("raw_data")
output_dir.mkdir(parents=True, exist_ok=True)


def valid_gzip(path):
    """Check whether a gzip file is readable to the end."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False


current = start
success = 0
failed = []

total = int((end - start).total_seconds() // 3600)

for i in range(total):
    filename = f"{current:%Y-%m-%d}-{current.hour}.json.gz"
    filepath = output_dir / filename
    temp_path = output_dir / (filename + ".part")

    print(f"[{i+1}/{total}] {filename}", flush=True)

    if valid_gzip(filepath):
        print("  Already downloaded")
        success += 1
    else:
        if filepath.exists():
            filepath.unlink()

        url = f"https://data.gharchive.org/{filename}"

        result = subprocess.run([
            "curl.exe",
            "-L",
            "--retry", "3",
            "--retry-delay", "2",
            "--connect-timeout", "30",
            "--fail",
            "--silent",
            "--show-error",
            "-o", str(temp_path),
            url
        ])

        if result.returncode == 0 and valid_gzip(temp_path):
            temp_path.replace(filepath)
            success += 1
            print("  Download OK")
        else:
            failed.append(filename)
            print("  Download FAILED")

        if temp_path.exists():
            temp_path.unlink()

    current += timedelta(hours=1)

print("\nDownload Summary")
print(f"Expected: {total}")
print(f"Successful: {success}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed files:")
    for name in failed:
        print(name)
else:
    print("All files downloaded successfully!")
