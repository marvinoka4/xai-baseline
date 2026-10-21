import hashlib

from src.config import DATASET, DROPPED, EXPECTED_ROWS, FEATURES, MISSING_LABEL
from src.data import expected_sha256, load, load_raw


def test_committed_snapshot_matches_recorded_sha256():
    recorded = expected_sha256()
    assert recorded is not None, "data/SNAPSHOT.md has no 'Snapshot SHA256' line"
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == recorded


def test_snapshot_has_all_rows_and_both_source_files():
    raw = load_raw()
    assert len(raw) == EXPECTED_ROWS
    assert raw["source_file"].value_counts().to_dict() == {"adult.data": 32561,
                                                            "adult.test": 16281}
    assert set(raw["income"]) == {">50K", "<=50K"}


def test_prepare_drops_columns_and_recodes_missing_values():
    X, y = load()
    assert list(X.columns) == FEATURES
    assert not set(DROPPED) & set(X.columns)
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 11687
    for col in X.select_dtypes("category"):
        assert "?" not in X[col].cat.categories
    assert (X["workclass"] == MISSING_LABEL).sum() == 2799
    assert (X["occupation"] == MISSING_LABEL).sum() == 2809
    assert (X["native_country"] == MISSING_LABEL).sum() == 857
    assert not X.isna().any().any()
