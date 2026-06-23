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
