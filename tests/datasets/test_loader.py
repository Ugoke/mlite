import numpy as np
import pytest

from mlite.datasets import available_datasets, load_dataset


def test_available_datasets() -> None:
    assert available_datasets() == ("student", "smart_city")


def test_load_student_dataset() -> None:
    dataset = load_dataset("student")

    assert isinstance(dataset, np.ndarray)
    assert dataset.shape == (10_000,)
    assert dataset.dtype.names == (
        "student_id",
        "age",
        "study_hours_per_week",
        "attendance_pct",
        "average_grade",
        "assignments_count",
        "sleep_hours_per_night",
        "final_score",
        "passed",
    )
    assert dataset["age"].dtype.kind in {"i", "u"}


def test_load_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("missing")
