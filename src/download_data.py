"""Download UCI Adult and write the committed snapshot data/adult.csv.gz.

The snapshot combines adult.data (32,561 rows) and adult.test (16,281 rows),
strips whitespace, removes the trailing "." from test labels, and adds a
`source_file` column. It is written deterministically (gzip with no file name
and mtime 0, LF line endings), so rebuilding it gives the same SHA256 as long
as UCI has not changed the files.

Usage: python -m src.download_data
"""

import csv
import gzip
import hashlib
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from src.config import COLUMNS, DATA_RAW, DATASET, EXPECTED_ROWS, RAW_ZIP, SOURCE_URL


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_member(archive, name):
    rows = []
    text = archive.read(name).decode("utf-8")
    for line in text.splitlines():
        if not line.strip() or line.startswith("|"):
            continue  # blank lines and the "|1x3 Cross validator" header in adult.test
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{name}: expected {len(COLUMNS)} fields, got {len(fields)}")
        fields[-1] = fields[-1].rstrip(".")
        rows.append(fields + [name])
    return rows


def build_snapshot(zip_bytes):
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    rows = read_member(archive, "adult.data") + read_member(archive, "adult.test")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(rows)}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS + ["source_file"])
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=9, mtime=0), rows


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "xai-baseline/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        zip_bytes = response.read()
    RAW_ZIP.write_bytes(zip_bytes)
    data, rows = build_snapshot(zip_bytes)
    DATASET.write_bytes(data)
    print(f"source_url: {SOURCE_URL}")
    print(f"downloaded_at_utc: {downloaded_at}")
    print(f"zip_sha256: {sha256_bytes(zip_bytes)}")
    print(f"rows: {len(rows)} (adult.data {sum(r[-1] == 'adult.data' for r in rows)}, "
          f"adult.test {sum(r[-1] == 'adult.test' for r in rows)})")
    print(f"snapshot: {DATASET.name} sha256 {sha256_bytes(data)}")


if __name__ == "__main__":
    main()
