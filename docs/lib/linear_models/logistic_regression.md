# LogisticRegression

`LogisticRegression` is a binary classification model that predicts probabilities using the logistic (sigmoid) function.

The model is trained using batch gradient descent optimization.

---

# Mathematical Model

The model computes a linear combination of input features:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

Where:

* `x` — input feature
* `w` — weight
* `b` — bias (intercept)
* `z` — linear output

The linear output is transformed into a probability using the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Predicted probability:

$$
P(y = 1 \mid x) = \sigma(z)
$$

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

The implementation optimizes Binary Cross Entropy (BCE) using its analytical derivative.

Binary Cross Entropy (BCE):

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n}
\left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
$$

Where:

* `y` — true label
* `p` — predicted probability

The implementation does not explicitly compute BCE during training.

Instead, it uses the derivative of BCE with respect to the linear output:

$$
\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y
$$

Which is mathematically equivalent for logistic regression with sigmoid activation and BCE loss.

---

# Creating a Model

```python
from mlite.linear_models import LogisticRegression

model = LogisticRegression(
    learning_rate=0.01,
    epochs=1000
)
```

| Parameter       | Type    |
| --------------- | ------- |
| `learning_rate` | `float` |
| `epochs`        | `int`   |

Requirements:

* `learning_rate` must be finite and positive
* `epochs` must be positive

---

# Training

```python
model.fit(
    X,
    y
)
```

| Parameter | Type       | Description                            |
| --------- | ---------- | -------------------------------------- |
| `X`       | array-like | Input matrix `(n_samples, n_features)` |
| `y`       | array-like | Binary labels `(n_samples,)`           |

Requirements:

* `X` must be non-empty
* `y` must be non-empty
* `X` and `y` must contain the same number of samples
* `y` must contain only binary labels `0` and `1`

One-dimensional input is automatically reshaped:

```python
X.shape == (n_samples,)
```

becomes

```python
X.shape == (n_samples, 1)
```

Training uses batch gradient descent:

1. Compute predictions for all samples
2. Accumulate gradients
3. Update weights once per epoch

---

# Predicting Probabilities

```python
probabilities = model.predict_proba(X)
```

Returns:

```python
np.ndarray with shape (n_samples,)
```

Values are in range:

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
np.ndarray with shape (n_samples,)
```

containing values:

```python
0 or 1
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

# Saving Model State

```python
state = model.state_dict()
```

Returns:

```python
{
    "coef": ...,
    "intercept": ...,
    "n_features_in": ...,
    "learning_rate": ...,
    "epochs": ...
}
```

---

# Loading Model State

```python
model.load_state_dict(state)
```

Restores model parameters and training configuration.

The state dictionary must contain:

* `coef`
* `intercept`
* `n_features_in`
* `learning_rate`
* `epochs`

---

# Accessing Parameters

```python
model.coef_
model.intercept_
model.n_features_in_
```

| Property         | Description                      |
| ---------------- | -------------------------------- |
| `coef_`          | Returns learned weights          |
| `intercept_`     | Returns learned bias             |
| `n_features_in_` | Returns number of input features |

---

# Example

```python
import numpy as np

from mlite.linear_models import LogisticRegression

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

model = LogisticRegression(
    learning_rate=0.001,
    epochs=5000
)

model.fit(X, y)

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
* Optimizes BCE through analytical derivatives
* Supports binary classification only
* Labels must be `0` or `1`
* Uses float64 computations
* Input normalization is not automatic
* Feature scaling may improve convergence
* Prediction threshold is configurable
* Supports model serialization through `state_dict()`
* Raises an exception when prediction methods are called before fitting
* BCE is not explicitly computed during training
