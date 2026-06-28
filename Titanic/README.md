# Titanic Survival Prediction — Logistic Regression

A machine learning model that predicts passenger survival on the Titanic using
logistic regression. Built for the classic
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

## Overview

The notebook [`77-titanic-prediction-logreg.ipynb`](77-titanic-prediction-logreg.ipynb)
loads the competition data, engineers a handful of features, tunes a logistic
regression model with cross-validation, and writes a `submission.csv` ready for
upload to Kaggle.

The tuned model reaches roughly **83% cross-validated accuracy** (10-fold).

## Data

The model expects the standard Kaggle Titanic files in the read-only input
directory:

- `train.csv` — labeled training data (includes the `Survived` target)
- `test.csv` — unlabeled test data for predictions
- `gender_submission.csv` — sample submission format

## Feature Engineering

| Feature | Description |
| --- | --- |
| `Title` | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare). Rare/duplicate titles are consolidated. |
| `FamilySize` | `SibSp + Parch + 1` — combines siblings/spouses and parents/children. |
| `IsAlone` | Flag set when `FamilySize == 1`. |
| `Sex`, `Embarked`, `Title` | One-hot encoded (with `drop_first=True`). |

Other preprocessing steps:

- Dropped columns: `Name`, `Ticket`, `Cabin`.
- Missing `Age` filled with the training median; missing `Fare` filled with the median.
- All features standardized with `StandardScaler`.

## Model

- **Algorithm:** `sklearn.linear_model.LogisticRegression` (`max_iter=1000`)
- **Hyperparameter tuning:** `GridSearchCV` over the regularization strength
  `C` ∈ {0.5, 1, 2, 5, 10} with 10-fold cross-validation, scored on accuracy.
- **Best parameter found:** `C = 0.5` (≈ 0.83 CV accuracy).

## Requirements

```
numpy
pandas
scikit-learn
kagglehub
```

## Usage

1. Open the notebook in a Kaggle environment (or any Jupyter environment with the
   data available under `/kaggle/input/competitions/titanic/`).
2. Run all cells. The notebook will:
   - Load and preprocess the data.
   - Tune and fit the logistic regression model.
   - Generate predictions on the test set.
   - Write `submission.csv` for Kaggle submission.

## Output

`submission.csv` with two columns: `PassengerId` and the predicted `Survived`
(0 or 1).
