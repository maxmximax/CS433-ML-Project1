"""Train the final model and write the AIcrowd submission file.

Usage (from the project1/ folder):
    python3 run.py

Pipeline:
    1. Load the data (cached as .npz after the first run, since parsing the CSVs is slow).
    2. Preprocess the features.
    3. Train the model and choose the decision threshold on a validation split.
    4. Predict the test set and write the submission.
"""

import os

import numpy as np

from helpers import create_csv_submission, load_csv_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset")
CACHE_PATH = os.path.join(DATA_PATH, "cache.npz")
SUBMISSION_PATH = "submission.csv"

SEED = 42  # fixed so that run.py always produces the same submission
VALIDATION_RATIO = 0.2  # fraction of the training set held out for validation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_path=DATA_PATH, cache_path=CACHE_PATH):
    """Load the dataset, caching it as a .npz file to avoid re-parsing the CSVs.

    Parsing the CSVs takes about a minute; loading the cache takes a few seconds.
    Delete the cache file to force a reload from the CSVs.

    Args:
        data_path: folder containing x_train.csv, y_train.csv and x_test.csv.
        cache_path: where to store the parsed arrays.

    Returns:
        x_train: numpy array of shape (N, D), raw training features (NaN = missing).
        x_test: numpy array of shape (M, D), raw test features.
        y_train: numpy array of shape (N,), labels in {-1, 1}.
        train_ids: numpy array of shape (N,), ids of the training samples.
        test_ids: numpy array of shape (M,), ids of the test samples.
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    np.savez(
        cache_path,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    return x_train, x_test, y_train, train_ids, test_ids


def train_validation_split(n, ratio=VALIDATION_RATIO, seed=SEED):
    """Randomly split the indices 0..n-1 into a train and a validation part.

    Args:
        n: number of samples.
        ratio: fraction of the samples used for validation.
        seed: random seed, for reproducibility.

    Returns:
        tr_idx, val_idx: numpy arrays of indices.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(ratio * n)
    return idx[n_val:], idx[:n_val]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Model and evaluation
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

