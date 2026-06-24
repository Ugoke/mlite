# LinearRegression

`LinearRegression` is a simple linear regression model trained using gradient descent.

---

# Mathematical Model

### The model predicts values using a linear equation:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

### where:
- x — input feature
- w — weight
- b — bias (intercept)
- y_pred — predicted value

# Loss Function

During training, the model minimizes the Mean Squared Error (MSE):

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


# Creating a Model

```python
from mlite.linear_models import LinearRegression

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)
```

| Parameter       | Type    |
| --------------- | ------- |
| `learning_rate` | `float` |
| `epochs`        | `int`   |


# Training

```python
model.fit(X, y)
```

| Parameter | Shape            |
| --------- | ---------------- |
| `X`       | `(n_samples, 1)` |
| `y`       | `(n_samples,)`   |


# Prediction

```python
predictions = model.predict(X)
```


# Score

```python
score = model.score(X, y)
```
Returns $R^2$ score.

| Score | Meaning                        |
| ----- | ------------------------------ |
| 1.0   | Perfect prediction             |
| 0.0   | Same as predicting the mean    |
| < 0.0 | Worse than predicting the mean |


# Saving Model State

```python
state = model.state_dict()
```

Returns a dictionary containing all parameters required to restore the trained model.

| Key              | Description                                   |
| ---------------- | --------------------------------------------- |
| `coef`           | Learned weights                               |
| `intercept`      | Learned bias                                  |
| `n_features_in`  | Number of input features                      |
| `learning_rate`  | Learning rate used during training            |
| `epochs`         | Number of training epochs                     |


# Loading Model State

```python
model.load_state_dict(state)
```

Restores model parameters from a previously saved state dictionary.

| Parameter | Type   | Description              |
| ---------- | ------ | ------------------------ |
| `state`    | `dict` | Model state dictionary   |


# Example

```python
import numpy as np

from mlite.linear_models import LinearRegression

X = np.array(
    [[1.0],
     [2.0],
     [3.0],
     [4.0],
     [5.0]],
    dtype=np.float64
)

y = np.array(
    [2.0, 4.0, 6.0, 8.0, 10.0],
    dtype=np.float64
)

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(
    np.array(
        [[6.0],
         [7.0]],
        dtype=np.float64
    )
)

print(predictions)
```
# Attributes

```python
model.coef_
model.intercept_
```

| Attribute      | Description                                   |
| -------------- | --------------------------------------------- |
| coef_          | Learned weights                               |
| intercept_     | Learned bias                                  |
| n_features_in_ | Number of input features used during training |



# Notes
- Uses batch gradient descent
- Uses float64 arrays
- Input normalization is not automatic
- Feature scaling may improve convergence