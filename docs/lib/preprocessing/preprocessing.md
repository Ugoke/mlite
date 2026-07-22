# Preprocessing

The `preprocessing` module converts categorical input into numeric matrices
that can be consumed by machine-learning models. It supports string and numeric
categories in one-dimensional or two-dimensional NumPy-compatible input.
Categories are learned independently for each feature and sorted in ascending
order. Missing values (`None` and `NaN`) are not supported.

---

## `OneHotEncoder`

`OneHotEncoder` creates one binary output column for every learned category.
For feature $j$ with categories $C_j = (c_{j,0}, \ldots, c_{j,k_j-1})$,
the encoded value is

$$
z_{i,j,r} = \mathbb{1}(x_{i,j} = c_{j,r}),
\qquad r = 0, \ldots, k_j - 1.
$$

The output groups features in input-column order. Within each group, columns
follow the order exposed by `categories_`.

### Construction

```python
OneHotEncoder(handle_unknown="error")
```

`handle_unknown` accepts:

* `"error"` (default): `transform` raises `ValueError` for a category not seen
  during `fit`;
* `"ignore"`: an unknown category becomes an all-zero group. Its inverse value
  is `None`.

### Methods and attributes

* `fit(X) -> None`: learns categories;
* `transform(X) -> np.ndarray`: returns a two-dimensional `float64` matrix;
* `fit_transform(X) -> np.ndarray`: fits and transforms in one call;
* `inverse_transform(X) -> np.ndarray`: restores original category values;
* `categories_`: a list containing the sorted categories of every feature;
* `n_features_in_`: number of input features seen during `fit`;
* `n_features_out_`: total number of one-hot columns;
* `state_dict()` and `load_state_dict(state)`: serialize and restore learned
  categories and configuration.

Learned attributes, `transform`, `inverse_transform`, and `state_dict` raise
`RuntimeError` before fitting. A different number of features raises
`ValueError`.

### Example

```python
import numpy as np
from mlite.preprocessing import OneHotEncoder

X = np.array([
    ["red", "S"],
    ["blue", "M"],
    ["red", "M"],
])

encoder = OneHotEncoder()
encoded = encoder.fit_transform(X)

print(encoder.categories_)
# [array(['blue', 'red']), array(['M', 'S'])]

print(encoded)
# [[0. 1. 0. 1.]
#  [1. 0. 1. 0.]
#  [0. 1. 1. 0.]]
```

---

## `OrdinalEncoder`

`OrdinalEncoder` replaces each category by its zero-based position in the
feature's sorted category list:

$$
z_{i,j} = r \quad \text{if} \quad x_{i,j} = c_{j,r}.
$$

The numerical order is a code only; it does not imply a meaningful distance
between categories.

### Construction

```python
OrdinalEncoder(handle_unknown="error", unknown_value=-1.0)
```

`handle_unknown` accepts:

* `"error"` (default): `transform` raises `ValueError` for unseen categories;
* `"use_encoded_value"`: unseen categories are represented by
  `unknown_value`. The value must be finite and must not collide with a learned
  zero-based code. Its inverse value is `None`.

### Methods and attributes

* `fit(X) -> None`: learns categories;
* `transform(X) -> np.ndarray`: returns a two-dimensional `float64` matrix;
* `fit_transform(X) -> np.ndarray`: fits and transforms in one call;
* `inverse_transform(X) -> np.ndarray`: restores original category values;
* `categories_`: a list containing the sorted categories of every feature;
* `n_features_in_`: number of input features seen during `fit`;
* `state_dict()` and `load_state_dict(state)`: serialize and restore learned
  categories and configuration.

### Example

```python
import numpy as np
from mlite.preprocessing import OrdinalEncoder

X = np.array([
    ["red", "S"],
    ["blue", "M"],
    ["red", "M"],
])

encoder = OrdinalEncoder()
encoded = encoder.fit_transform(X)

print(encoded)
# [[1. 1.]
#  [0. 0.]
#  [1. 0.]]

original = encoder.inverse_transform(encoded)
```

Both encoders replace their learned state when `fit` is called again. State
dictionaries contain independent category-array copies and can be loaded into
a newly constructed encoder.
