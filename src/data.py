"""Load the committed Adult snapshot and prepare features, as fixed in PROTOCOL.md."""

import hashlib
import re

import pandas as pd

from src.config import (CATEGORICAL, DATASET, DROPPED, FEATURES, MISSING_LABEL, NUMERIC,
                        SNAPSHOT_MD, TARGET)


def expected_sha256():
    """The snapshot SHA256 recorded in data/SNAPSHOT.md."""
    match = re.search(r"Snapshot SHA256: `([0-9a-f]{64})`", SNAPSHOT_MD.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def load_raw(check_hash=True):
    data = DATASET.read_bytes()
    if check_hash:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256():
            raise SystemExit(f"Snapshot hash mismatch: {digest} != {expected_sha256()}")
    return pd.read_csv(DATASET, dtype=str, keep_default_na=False)


def prepare(raw):
    """Target, feature types and missing values. No statistics are learned here,
    so this step cannot leak information across folds."""
    frame = raw.drop(columns=DROPPED + ["source_file"])
    labels = frame.pop("income")
    unexpected = set(labels) - {">50K", "<=50K"}
    if unexpected:
        raise ValueError(f"Unexpected labels: {unexpected}")
    y = (labels == ">50K").astype(int).rename(TARGET)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    for col in CATEGORICAL:
        # "?" marks a missing value in the UCI files; keep it as its own category.
        frame[col] = frame[col].replace("?", MISSING_LABEL).astype("category")
    return frame[FEATURES], y


def load():
    return prepare(load_raw())
