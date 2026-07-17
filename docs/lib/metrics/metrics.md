# Metrics
The `metrics` module provides functions for evaluating machine learning model performance.
Metrics are used after training a model to measure how accurately it predicts new data. The module supports evaluation for both classification and regression tasks.

---

## `accuracy_score`
Calculates the accuracy of a classification model.
Accuracy represents the ratio of correctly predicted samples to the total number of samples.

Formula:

```python
accuracy = correct_predictions / total_predictions
```

Returns: `float` value in range `0.0 - 1.0`.
* `y_true`: true class labels
* `y_pred`: predicted class labels

### Example

```python
from mlite.metrics import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0], dtype=np.float64)
y_pred = np.array([0, 1, 0, 0], dtype=np.float64)

score = accuracy_score(y_true, y_pred)

print(score)
```

---

## `r2_score`

Calculates the coefficient of determination (R² score) for regression models.
R² measures how well predicted values explain the variance of the target values.

Formula:

```
R² = 1 - (SS_residual / SS_total)
```

Returns: `float` value.

The score interpretation:
* `1.0`: perfect predictions
* `0.0`: predictions are equal to the mean baseline
* `< 0.0`: model performs worse than the baseline
* `y_true`: true target values
* `y_pred`: predicted target values

### Example

```python
from mlite.metrics import r2_score
import numpy as np

y_true = np.array([1, 2, 3, 4], dtype=np.float64)
y_pred = np.array([1, 2, 3, 3], dtype=np.float64)

score = r2_score(y_true, y_pred)

print(score)
```

---
