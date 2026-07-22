from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..utils._constants import DATASETS


def available_datasets() -> Tuple[str, ...]:
    """Return the names accepted by :func:`load_dataset`."""

    return tuple(DATASETS)


def load_dataset(name: str) -> NDArray[np.void]:
    """Load a bundled CSV dataset into a NumPy structured array.

    Parameters
    ----------
    name : str
        Dataset name: ``"student"`` or ``"smart_city"``.

    Returns
    -------
    np.ndarray
        A one-dimensional structured array. CSV headers become named fields, so
        values are available by column name, for example ``data["age"]``.

    Raises
    ------
    ValueError
        If *name* is not the name of a bundled dataset.
    """

    try:
        filename = DATASETS[name]
    except KeyError as error:
        choices = ", ".join(available_datasets())
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {choices}.") from error

    path = Path(__file__).with_name("data") / filename
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
