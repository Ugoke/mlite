# Model Selection

The `model_selection` module provides utilities for preparing datasets before training machine learning models. It allows splitting data into training and test sets, which helps measure how well a model performs on unseen data.

Proper dataset splitting is an important step in machine learning because it prevents evaluating a model on the same data it was trained on. The module also provides reproducible splits using a random seed and supports stratified splitting to preserve class distribution in classification tasks.

## `train_test_split`

Splits dataset into training and test sets by shuffling and dividing according to `test_size`.

Returns: `X_train, X_test, y_train, y_test`.

* `X`: feature matrix
* `y`: target vector
* `test_size`: fraction of test data (0–1)
* `random_state`: seed for reproducibility

### Example

```python
from mlite.model_selection import train_test_split
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
y = np.array([0, 0, 1, 1], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42
)

print(X_train)
print(X_test)
```

---

## `stratified_shuffle_split`

Splits a dataset into training and test sets while preserving the class distribution in both subsets.

Returns: `X_train, X_test, y_train, y_test`.

* `X`: feature matrix
* `y`: target vector
* `test_size`: fraction of test data (0–1)
* `random_state`: seed for reproducibility

### Example

```python
from mlite.model_selection import stratified_shuffle_split
import numpy as np

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12]
], dtype=np.float64)

y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

X_train, X_test, y_train, y_test = stratified_shuffle_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

print(X_train)
print(X_test)
```

---

## `k_fold`

Splits a dataset into n_splits consecutive folds for k-fold cross-validation.
During each iteration one fold is used as validation data, while the remaining folds are combined into the training data.

Returns: `list[(X_train, X_test, y_train, y_test)]`
Each element contains one train/test split.

Parameters
* `X`: feature matrix with shape `(n_samples, n_features)`
* `y`: target vector with shape `(n_samples,)`
* `n_splits`: number of folds
* `shuffle`: whether to shuffle samples before splitting
* `random_state`: random seed used when `shuffle=True`

### Example
```python
from mlite.model_selection import k_fold
import numpy as np


X = np.arange(20, dtype=np.float64).reshape(10, 2)
y = np.arange(10, dtype=np.float64)


folds = k_fold(
    X,
    y,
    n_splits=5,
    shuffle=True,
    random_state=42
)


for X_train, X_test, y_train, y_test in folds:
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
```

### Notes
* `n_splits` must be greater than 1.
* `n_splits` cannot exceed the number of samples.
* Every sample appears exactly once in a validation set.
* When `shuffle=False`, splitting order is deterministic.

---

## `cross_val_score`

Evaluates an estimator using k-fold cross-validation.

For every fold:
* creates a new unfitted copy of the estimator;
* trains the estimator on the training subset;
* evaluates the estimator on the validation subset;
* stores the resulting score.
The original estimator is not modified.

Returns: `np.ndarray`
containing one score per fold.

### Parameters
* `estimator`: model implementing:
    * `fit(X, y)`
    * `score(X, y)`
* `X`: feature matrix
* `y`: target vector
* `n_splits`: number of cross-validation folds
* `shuffle`: whether to shuffle before splitting
* `random_state`: random seed

### Example
```python
from mlite.model_selection import cross_val_score
from mlite.linear_models import LinearRegression
import numpy as np


X = np.array(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
    ],
    dtype=np.float64
)

y = np.array(
    [2, 4, 6, 8, 10, 12],
    dtype=np.float64
)


model = LinearRegression()


scores = cross_val_score(
    model,
    X,
    y,
    n_splits=3,
    shuffle=True,
    random_state=42
)


print(scores)
print(scores.mean())
```

### Reproducibility
All functions that use randomness accept `random_state`.

Using the same seed produces identical splits:
```python
scores1 = cross_val_score(model, X, y, random_state=42)
scores2 = cross_val_score(model, X, y, random_state=42)

assert np.array_equal(scores1, scores2)
```

### Limitations
* Input arrays must contain numeric values convertible to `float64`.
* `X` must be one-dimensional or two-dimensional.
* `y` must be one-dimensional.
* The number of samples in `X` and `y` must match.
* `cross_val_score` requires estimators implementing `fit` and `score`.