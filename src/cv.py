"""Stratified 5-fold cross-validation repeated over 5 split seeds (25 folds)."""

from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, SPLIT_SEEDS


def folds(y, seeds=SPLIT_SEEDS, n_splits=N_SPLITS):
    """Yield (seed, fold_index, train_positions, test_positions) for every fold."""
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (train, test) in enumerate(splitter.split(y, y)):
            yield seed, k, train, test
