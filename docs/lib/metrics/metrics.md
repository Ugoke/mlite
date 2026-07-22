# Metrics
The `metrics` module provides functions for evaluating machine learning model performance.
Metrics are used after training a model to measure how accurately it predicts new data. The module supports evaluation for both classification and regression tasks.

---

## `accuracy_score`
Calculates the accuracy of a classification model.
Accuracy represents the ratio of correctly predicted samples to the total number of samples.

Formula:

$$
\operatorname{Accuracy} = \frac{1}{n} \sum_{i=1}^{n}
\mathbb{1}(y_i = \hat{y}_i)
$$

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

## `mean_absolute_error`

Calculates the mean absolute error (MAE) for regression models. MAE is the
average absolute difference between target values and predictions; lower
values indicate better predictions.

Formula:

$$
\operatorname{MAE} = \frac{1}{n} \sum_{i=1}^{n}
\left|y_i - \hat{y}_i\right|
$$

Returns: a non-negative `float`.

* `y_true`: true target values
* `y_pred`: predicted target values

### Example

```python
from mlite.metrics import mean_absolute_error
import numpy as np

y_true = np.array([1, 2, 3, 4], dtype=np.float64)
y_pred = np.array([1, 3, 2, 5], dtype=np.float64)

error = mean_absolute_error(y_true, y_pred)

print(error)  # 0.75
```

---

## `recall_score`

Calculates recall for binary classification, using `1.0` as the positive
class. Recall is the proportion of actual positive samples identified as
positive.

Formula:

$$
\operatorname{Recall} = \frac{TP}{TP + FN}
$$

Returns: a `float` in the range `0.0` to `1.0`. If `y_true` contains no
positive samples, returns `0.0`.

* `y_true`: true binary labels
* `y_pred`: predicted binary labels

### Example

```python
from mlite.metrics import recall_score
import numpy as np

y_true = np.array([1, 1, 0, 0], dtype=np.float64)
y_pred = np.array([1, 0, 1, 0], dtype=np.float64)

score = recall_score(y_true, y_pred)

print(score)  # 0.5
```

---

## `root_mean_squared_error`

Calculates the root mean squared error (RMSE) for regression models. RMSE is
the square root of MSE, so it is expressed in the same units as the target
values; lower values indicate better predictions.

Formula:

$$
\operatorname{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}
\left(y_i - \hat{y}_i\right)^2}
$$

Returns: a non-negative `float`.

* `y_true`: true target values
* `y_pred`: predicted target values

### Example

```python
from mlite.metrics import root_mean_squared_error
import numpy as np

y_true = np.array([1, 2, 3, 4], dtype=np.float64)
y_pred = np.array([1, 2, 2, 5], dtype=np.float64)

error = root_mean_squared_error(y_true, y_pred)

print(error)  # approximately 0.707
```

---

## `mean_squared_error`

Calculates the mean squared error (MSE) for regression models. MSE is the
average of the squared differences between target values and predictions;
lower values indicate better predictions.

Formula:

$$
\operatorname{MSE} = \frac{1}{n} \sum_{i=1}^{n}
\left(y_i - \hat{y}_i\right)^2
$$

Returns: a non-negative `float`.

* `y_true`: true target values
* `y_pred`: predicted target values

### Example

```python
from mlite.metrics import mean_squared_error
import numpy as np

y_true = np.array([1, 2, 3, 4], dtype=np.float64)
y_pred = np.array([1, 2, 2, 5], dtype=np.float64)

error = mean_squared_error(y_true, y_pred)

print(error)  # 0.5
```

---

## `r2_score`

Calculates the coefficient of determination (R² score) for regression models.
R² measures how well predicted values explain the variance of the target values.

Formula:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2}
{\sum_{i=1}^{n}\left(y_i - \bar{y}\right)^2}
$$

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
