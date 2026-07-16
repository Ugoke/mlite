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