# CS-433 Project 1: Predicting coronary heart disease (MICHD)

Binary classification on the BRFSS dataset: predict whether a respondent has
coronary heart disease (MICHD) from 321 health and lifestyle features.
Only Python's standard library and NumPy are used.

## Repository structure

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`) and their loss/gradient helpers. |
| `run.py` | Full pipeline: loading, preprocessing, training, threshold selection, and generation of `submission.csv`. |
| `helpers.py` | Course-provided data loading and submission helpers. |
| `grading_tests/` | Course-provided public tests. |
| `latex-example-paper/` | LaTeX template for the report. |

## Setup

1. Install Python 3 with NumPy.
2. Download the data from the
   [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
   and put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/dataset/`
   (the `data/` folder is not tracked by git because of its size).

## Data

- `x_train.csv`: 328,135 samples x 321 features, `x_test.csv`: 109,379 samples.
  Missing values are read as `NaN`.
- `y_train.csv`: labels in {-1, 1} (1 = MICHD), with about 8.8% positives, so the
  classes are strongly imbalanced.
- `run.py` loads the CSVs with `helpers.load_csv_data` and caches the parsed arrays in
  `data/dataset/cache.npz`. The first run takes about a minute, later runs a few
  seconds. Delete the cache to force a reload.
- 20% of the training set is held out for validation, with a fixed seed (`SEED = 42`)
  so the split and the submission are reproducible. Both values are set in the
  configuration section at the top of `run.py`.

## Reproducing the submission

```bash
python3 run.py
```

This writes `submission.csv`.

## Preprocessing


## Model


## Results


## Running the tests

```bash
cd grading_tests
python3 -m pip install pytest gitpython black
python3 -m pytest --github_link .. . -q
```
