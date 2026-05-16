# LinearRegression

`LinearRegression` is a simple linear regression model trained using gradient descent.

---

# Mathematical Model

The model is defined by the following linear function:

### ŷ = wx + b

where:

* `x` — input feature
* `w` — model weight
* `b` — bias
* `ŷ` — predicted value

---

# Loss Function

During training, the model minimizes the Mean Squared Error (MSE):

## MSE = (1 / n) * Σ(y - ŷ)^2

---

# Creating a Model

```python
from mlite import LinearRegression

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)
```

| Parameter       | Type    |
| --------------- | ------- |
| `learning_rate` | `float` |
| `epochs`        | `int`   |

---

# Training

```python
model.fit(X, y)
```

| Parameter | Shape            |
| --------- | ---------------- |
| `X`       | `(n_samples, 1)` |
| `y`       | `(n_samples,)`   |

---

# Prediction

```python
predictions = model.predict(X)
```

---

# Example

```python
import numpy as np
from mlite import LinearRegression

X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(
    np.array([6, 7], dtype=np.float64)
)

print(predictions)
```

---

# Notes

The gradient is calculated without the `2` coefficient from the MSE derivative.
This is not an error, since the constant factor is typically absorbed into the `learning_rate`.
