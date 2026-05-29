# LogisticRegression
`LogisticRegression` is a binary classification model that predicts probabilities using the logistic (sigmoid) function.
The model is trained using batch gradient descent optimization.
---
# Mathematical Model
The model computes a linear combination of input features:
## z = w1x1 + w2x2 + ... + wn*xn + b
Where:
* `x` — input feature
* `w` — weight
* `b` — bias (intercept)
* `z` — linear output

The linear output is transformed into a probability using the sigmoid function:

## σ(z) = 1 / (1 + e^(-z))
Predicted probability:
## P(y = 1 | x) = σ(z)
Classification rule:

```python
prediction = 1 if probability >= threshold else 0
```

Default threshold:

```python
threshold = 0.5
```

---

# Loss Function

The implementation optimizes Binary Cross Entropy (BCE) using its analytical gradient.

Binary Cross Entropy:

## BCE = -(1 / n) * Σ[y * log(p) + (1 - y) * log(1 - p)]

Where:

* `y` — true label
* `p` — predicted probability

The implementation does not explicitly compute BCE during training.
Instead, it uses the simplified gradient:

## gradient = prediction - target

Which is mathematically equivalent for logistic regression with sigmoid activation and BCE loss.

---

# Creating a Model

```python
from mlite.linear_models.logistic_regression import LogisticRegression

model = LogisticRegression()
```

The model constructor initializes:

* empty coefficients
* intercept = 0.0
* n_features_in_ = 0

Training parameters are passed directly into `fit()`.

---

# Training

```python
model.fit(
    X,
    y,
    learning_rate=0.001,
    epochs=1000
)
```

| Parameter       | Type       | Description                            |
| --------------- | ---------- | -------------------------------------- |
| `X`             | array-like | Input matrix `(n_samples, n_features)` |
| `y`             | array-like | Binary labels `(n_samples,)`           |
| `learning_rate` | `float`    | Gradient descent step size             |
| `epochs`        | `int`      | Number of training iterations          |

Requirements:

* `y` must contain only:

  * `0`
  * `1`

Training uses batch gradient descent:

1. Compute predictions for all samples
2. Accumulate gradients
3. Update weights once per epoch

---

# Predicting Probabilities

```python
probabilities = model.predict_proba(X)
```

Returns probabilities in range:

```python
[0.0, 1.0]
```

| Value | Meaning                        |
| ----- | ------------------------------ |
| `0.0` | Probability of class 1 is 0%   |
| `1.0` | Probability of class 1 is 100% |

---

# Predicting Classes

```python
predictions = model.predict(X)
```

Returns:

```python
[0, 1]
```

Custom threshold:

```python
predictions = model.predict(
    X,
    threshold=0.7
)
```

| Parameter   | Type    |
| ----------- | ------- |
| `threshold` | `float` |

Threshold range:

```python
0.0 <= threshold <= 1.0
```

---

# Accuracy Score

```python
score = model.score(X, y)
```

Returns classification accuracy.

Accuracy formula:

## accuracy = correct_predictions / total_predictions

| Score | Meaning                |
| ----- | ---------------------- |
| `1.0` | Perfect classification |
| `0.5` | Random guessing level  |
| `0.0` | Completely incorrect   |

Custom threshold:

```python
score = model.score(
    X,
    y,
    threshold=0.7
)
```

---

# Loading Model State

```python
model.load_state(
    coef,
    intercept,
    n_features_in
)
```

Restores model parameters.

| Parameter       | Type       | Description              |
| --------------- | ---------- | ------------------------ |
| `coef`          | array-like | Learned weights          |
| `intercept`     | `float`    | Bias term                |
| `n_features_in` | `int`      | Number of input features |

---

# Accessing Parameters

```python
model.get_coef()
model.get_intercept()
model.get_n_features_in()
```

| Method                | Description                      |
| --------------------- | -------------------------------- |
| `get_coef()`          | Returns learned weights          |
| `get_intercept()`     | Returns learned bias             |
| `get_n_features_in()` | Returns number of input features |

---

# Example

```python
import numpy as np

from mlite.linear_models.logistic_regression import LogisticRegression

X = np.array(
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ],
    dtype=np.float64
)

y = np.array(
    [
        0,
        0,
        0,
        1,
        1,
        1,
    ],
    dtype=np.float64
)

model = LogisticRegression()

model.fit(
    X,
    y,
    learning_rate=0.001,
    epochs=5000
)

probabilities = model.predict_proba(X)
predictions = model.predict(X)
accuracy = model.score(X, y)

print(probabilities)
print(predictions)
print(accuracy)
```

---

# Notes

* Uses binary logistic regression
* Uses batch gradient descent
* Uses sigmoid activation
* Optimizes BCE through analytical gradients
* Supports binary classification only
* Labels must be `0` or `1`
* Uses float64 computations
* Input normalization is not automatic
* Feature scaling may improve convergence
* Prediction threshold is configurable
* Training parameters are not stored in the model state
* BCE is not explicitly computed during training
