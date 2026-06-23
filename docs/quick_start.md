# Quick Start

Train a simple linear regression model and make predictions.

```python
import numpy as np

from mlite.linear_models import LinearRegression

X = np.array([
    [1.0],
    [2.0],
    [3.0],
])

y = np.array([
    2.0,
    4.0,
    6.0,
])

model = LinearRegression()

model.fit(X, y)

pred = model.predict(
    np.array([
        [4.0],
        [5.0],
    ])
)

print(pred)
```
## Notes
- X must be a NumPy array with shape (n_samples, n_features)
- float64 arrays are recommended