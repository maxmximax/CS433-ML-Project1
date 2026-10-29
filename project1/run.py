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
import sys
import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import*

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data", "dataset")
CACHE_PATH = os.path.join(DATA_PATH, "cache.npz")
SUBMISSION_PATH = os.path.join(BASE_PATH, "submission.csv")

SEED = 42  # fixed so that run.py always produces the same submission
VALIDATION_RATIO = 0.2  # fraction of the training set held out for validation

LAMBDA = 1e-4  # regularization strength
GAMMA = 0.5    # learning rate for gradient descent
MAX_ITERS_TUNING = 2000
MAX_ITERS = 2000
LAMBDAS = [1e-4, 1e-3, 1e-2]   # grid used by --tune
GAMMAS = [0.5, 0.8 ,0.05, 0.1, 0.2]   # grid used by --tune 

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_path=DATA_PATH, cache_path=CACHE_PATH):
    """Load the dataset, caching it as a .npz file to avoid re-parsing the CSVs.

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


def train_validation_split(y, ratio=VALIDATION_RATIO, seed=SEED):
    """Create reproducible training and validation indices by class.

    Args:
        y: One-dimensional array of class labels, shape (N,).
        ratio: Fraction of each class allocated to validation, between
            0 and 1. Each class's validation count is rounded down.
        seed: Random seed controlling the shuffling.

    Returns:
        train_indices: One-dimensional array of training indices.
        validation_indices: One-dimensional array of validation indices.

    """
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []

    # Split each class separately to ensure that the training and validation
    # sets have the same class proportions as the original dataset.
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)

        n_validation = int(ratio * len(indices))
        validation_parts.append(indices[:n_validation])
        train_parts.append(indices[n_validation:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)

    # Shuffle the training and validation indices to avoid any ordering bias.
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return train_indices, validation_indices


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def fit_preprocessing(x):
    """Compute per-feature means and population standard deviations.

    Args:
        x: Raw feature matrix, shape (N, D).

    Returns:
        mean: Per-feature means, shape (D,).
        std: Per-feature standard deviations, shape (D,), with zero
            standard deviations replaced by 1.
    """
    # Compute the mean and standard deviation of each feature, ignoring
    # nonfinite values..
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    safe_counts = np.maximum(counts, 1)

    # Replace nonfinite values with zero to avoid NaNs in the mean and std.
    values = np.where(finite, x, 0.0)

    # Compute the mean of each feature, ignoring nonfinite values. 
    mean = values.sum(axis=0) / safe_counts

    # Compute the population standard deviation of each feature, ignoring
    # nonfinite values.
    centered = np.where(finite, x - mean, 0.0)
    std = np.sqrt(np.sum(centered**2, axis=0) / safe_counts)

    # Replace any zero standard deviations with 1 to avoid division by zero.
    std[std == 0] = 1.0

    return mean, std


def transform_features(x, mean, std):
    """Standardize features, replace nonfinite values, and add an intercept.

    Args:
        x: Raw feature matrix, shape (N, D).
        mean: Previously fitted per-feature means, shape (D,).
        std: Previously fitted, strictly positive per-feature standard
            deviations, shape (D,).

    Returns:
        Transformed feature matrix, shape (N, D + 1), whose first column
        contains ones.
    """
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return np.c_[np.ones(x.shape[0]), x]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred):
    """Compute binary classification metrics with 1 as the positive class.

    Args:
        y_true: True labels in {-1, 1}, shape (N,).
        y_pred: Predicted labels in {-1, 1}, shape (N,), aligned with
            y_true. Both arrays must be nonempty.

    Returns:
        Dictionary containing:
            accuracy: Fraction of correctly classified samples.
            precision: Fraction of predicted positives that are positive.
            recall: Fraction of actual positives predicted as positive.
            f1: Harmonic mean of precision and recall.

        All metrics range from 0 to 1. Precision, recall and F1 are set
        to zero when their respective denominators are zero.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    return {
        "accuracy": np.mean(y_pred == y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(name, y_true, y_pred):
    """Print classification metrics and positive-class proportions.

    Args:
        name: Heading identifying the dataset or evaluation configuration.
        y_true: True labels in {-1, 1}, shape (N,).
        y_pred: Predicted labels in {-1, 1}, shape (N,), aligned with
            y_true. Both arrays must be nonempty.

    Returns:
        None.

    Notes:
        Prints accuracy and positive-class proportions as percentages.
        Precision, recall and F1 are printed on a scale from 0 to 1.
    """
    metrics = classification_metrics(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy:  {100 * metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"Actual positives:    {100 * np.mean(y_true == 1):.2f}%")
    print(f"Predicted positives: {100 * np.mean(y_pred == 1):.2f}%")


def choose_threshold(y_true, probabilities):
    """Select the probability threshold maximizing validation F1.

    Args:
        y_true: Validation labels in {-1, 1}, shape (N,).
        probabilities: Finite predicted probabilities in [0, 1],
            shape (N,), aligned with y_true. N must be positive.

    Returns:
        threshold: Probability cutoff achieving the highest validation F1.
        best_f1: F1 score obtained with that cutoff.

    """
    order = np.argsort(-probabilities)
    sorted_probabilities = probabilities[order]
    sorted_positive = (y_true[order] == 1)

    true_positives = np.cumsum(sorted_positive)
    predicted_positives = np.arange(1, len(y_true) + 1)
    total_positives = np.sum(sorted_positive)

    # Evaluate only after all samples with an identical probability.
    boundaries = np.r_[
        np.flatnonzero(
            sorted_probabilities[:-1] != sorted_probabilities[1:]
        ),
        len(y_true) - 1,
    ]

    denominators = predicted_positives + total_positives
    f1_scores = 2 * true_positives / denominators

    best_index = boundaries[np.argmax(f1_scores[boundaries])]
    threshold = sorted_probabilities[best_index]

    return float(threshold), float(f1_scores[best_index])


def tune_hyperparameters(x_tr, y_tr, x_val, y_val_pm):
    """Grid search over (lambda, gamma), scored by best validation F1.

    Args:
        x_tr: Transformed training features, shape (N, D + 1).
        y_tr: Training labels in {0, 1}, shape (N,).
        x_val: Transformed validation features.
        y_val_pm: Validation labels in {-1, 1}.

    Returns:
        (best_lambda, best_gamma), the pair with the highest validation F1.
    """
    results = {}
    for lambda_ in LAMBDAS:
        for gamma in GAMMAS:
            w, _ = reg_logistic_regression(
                y_tr, x_tr, lambda_, np.zeros(x_tr.shape[1]), MAX_ITERS_TUNING, gamma
            )
            p_val = sigmoid(x_val @ w)
            if np.all(np.isfinite(p_val)):
                _, f1 = choose_threshold(y_val_pm, p_val)
            else:  # training diverged
                f1 = 0.0
            results[(lambda_, gamma)] = f1
            print(f"lambda={lambda_:g}  gamma={gamma:g}  F1={f1:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest: lambda={best[0]:g}, gamma={best[1]:g}, F1={results[best]:.4f}")
    return best
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run validation, threshold selection, final training, and submission.

    Load the dataset and verify that training labels are in {-1, 1}.
    Convert these labels to {0, 1} for logistic regression, then create
    a reproducible stratified training/validation split.

    Fit preprocessing on the training split and train a regularized
    logistic regression model using the configured hyperparameters.
    Report validation performance at the default probability threshold
    of 0.5, select a threshold maximizing validation F1, and report
    training and validation performance at the selected threshold.

    Retrain from zero weights on all labeled samples while retaining
    the preprocessing statistics fitted on the original training split.
    Apply the selected threshold to the final model's test probabilities
    and write predictions in {-1, 1} to SUBMISSION_PATH.

    Args:
        None. Paths and hyperparameters are read from module constants.

    Returns:
        None.

    Raises:
        ValueError: If any training label is outside {-1, 1}.

    """
    x_train_raw, x_test_raw, y_train, _, test_ids = load_data()
    y_train = np.asarray(y_train).reshape(-1)

    if not np.all(np.isin(y_train, [-1, 1])):
        raise ValueError("Expected training labels in {-1, 1}.")

    y_binary = (y_train == 1).astype(float)

    tr_idx, val_idx = train_validation_split(y_train)

    # Fit preprocessing using only the training split.
    mean, std = fit_preprocessing(x_train_raw[tr_idx])

    x_tr = transform_features(x_train_raw[tr_idx], mean, std)
    x_val = transform_features(x_train_raw[val_idx], mean, std)

    # `python3 run.py --tune`: search lambda and gamma, then stop.
    if "--tune" in sys.argv:
        tune_hyperparameters(x_tr, y_binary[tr_idx], x_val, y_train[val_idx])
        return


    print("Training the validation model...")

    w, loss = reg_logistic_regression(
        y=y_binary[tr_idx],
        tx=x_tr,
        lambda_=LAMBDA,
        initial_w=np.zeros(x_tr.shape[1]),
        max_iters=MAX_ITERS,
        gamma=GAMMA,
    )

    p_tr = sigmoid(x_tr @ w)
    p_val = sigmoid(x_val @ w)

    # Compare the default threshold with the F1-selected threshold.
    print_metrics(
        "Validation with threshold 0.5",
        y_train[val_idx],
        np.where(p_val >= 0.5, 1, -1),
    )

    threshold, best_f1 = choose_threshold(y_train[val_idx], p_val)

    print(f"\nSelected probability threshold: {threshold:.6f}")

    print_metrics(
        "Training with selected threshold",
        y_train[tr_idx],
        np.where(p_tr >= threshold, 1, -1),
    )

    print_metrics(
        "Validation with selected threshold",
        y_train[val_idx],
        np.where(p_val >= threshold, 1, -1),
    )

    # Retrain on all labeled samples.
    # Keep the same preprocessing used during threshold selection.
    del x_tr, x_val

    x_full = transform_features(x_train_raw, mean, std)

    print("\nRetraining on all labeled data...")

    w_final, final_loss = reg_logistic_regression(
        y=y_binary,
        tx=x_full,
        lambda_=LAMBDA,
        initial_w=np.zeros(x_full.shape[1]),
        max_iters=MAX_ITERS,
        gamma=GAMMA,
    )

    del x_full

    x_test = transform_features(x_test_raw, mean, std)
    p_test = sigmoid(x_test @ w_final)
    y_pred_test = np.where(p_test >= threshold, 1, -1)

    create_csv_submission(test_ids, y_pred_test, SUBMISSION_PATH)

    print(f"\nSubmission saved to: {SUBMISSION_PATH}")
    print(
        "Test samples predicted positive: "
        f"{100 * np.mean(y_pred_test == 1):.2f}%"
    )


if __name__ == "__main__":
    main()