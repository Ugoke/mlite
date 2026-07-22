# Datasets

The `mlite.datasets` module loads CSV datasets included with the package. It is
useful for experimenting with models without downloading data separately.

## Usage

```python
from mlite.datasets import available_datasets, load_dataset

print(available_datasets())

students = load_dataset("student")
ages = students["age"]
scores = students["final_score"]
```

`load_dataset(name)` accepts `"student"` and `"smart_city"`. Use
`available_datasets()` to obtain the current list.

## Result

The function returns a one-dimensional NumPy structured array. Each CSV column
is exposed as a named field, for example `students["age"]`. Numeric columns
retain numeric dtypes; text, date, and categorical columns remain strings.

`student` contains 10,000 student records, including `final_score` and `passed`.
`smart_city` contains 13,440 urban-infrastructure observations with traffic,
weather, parking, energy, camera, and IoT data.
