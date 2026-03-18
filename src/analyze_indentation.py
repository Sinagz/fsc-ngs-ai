from pathlib import Path
import json
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_JSON = BASE_DIR / "data" / "extracted" / "raw_layout" / "layout_lines.json"


def bucket_x0(x0, bucket_size=5):
    return round(x0 / bucket_size) * bucket_size


def main():
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {LAYOUT_JSON}")

    data = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))

    x0_counter = Counter()
    sample_lines_by_bucket = {}

    for page in data:
        for line in page["lines"]:
            text = line.get("text", "").strip()
            x0 = line.get("x0")

            if not text or x0 is None:
                continue

            bucket = bucket_x0(x0, bucket_size=5)
            x0_counter[bucket] += 1

            if bucket not in sample_lines_by_bucket:
                sample_lines_by_bucket[bucket] = []

            if len(sample_lines_by_bucket[bucket]) < 5:
                sample_lines_by_bucket[bucket].append(text)

    print("\n=== INDENTATION BUCKETS (x0 grouped by 5) ===\n")
    for bucket, count in sorted(x0_counter.items(), key=lambda x: x[0]):
        print(f"x0 ~ {bucket:>4}: {count:>5} lines")
        for sample in sample_lines_by_bucket[bucket]:
            print(f"   - {sample}")
        print()


if __name__ == "__main__":
    main()