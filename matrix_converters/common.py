import numpy as np
import pandas as pd
from pkg_resources import parse_version

LEGACY_PANDAS = parse_version(pd.__version__) < parse_version('0.24')


def coerce_matrix(matrix, allow_raw=True, force_square=True):
    """Infers a NumPy array from given input.

    Args:
        matrix (DataFrame, Series, ndarray or Iterable):
        allow_raw (bool, optional): Defaults to ``True``.
        force_square (bool, optional): Defaults to ``True``.

    Returns:
        ndarray: A 2D `np.ndarray` of type `float32`
    """
    if isinstance(matrix, pd.DataFrame):
        if force_square:
            assert matrix.index.equals(matrix.columns)
        matrix_values = matrix.values if LEGACY_PANDAS else matrix.to_numpy(copy=True)
        return matrix_values.astype(np.float32)
    elif isinstance(matrix, pd.Series):
        assert matrix.index.nlevels == 2, "Cannot infer a matrix from a Series with more or fewer than 2 levels"
        wide = matrix.unstack()

        union = wide.index | wide.columns
        wide = wide.reindex_axis(union, fill_value=0.0, axis=0).reindex_axis(union, fill_value=0.0, axis=1)
        wide = wide.values if LEGACY_PANDAS else wide.to_numpy(copy=True)
        return wide.astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    assert len(matrix.shape) == 2

    if force_square:
        i, j = matrix.shape
        assert i == j

    return matrix


def expand_array(a, n, axis=None):
    """Expands an array across all dimensions by a set amount.

    Args:
        a: The array to expand
        n: The (non-negative) number of items to expand by.
        axis (int or None, optional): The axis to expand along, or ``None`` to expand along all axes

    Returns:
        array: The expanded array
    """

    if axis is None:
        new_shape = [dim + n for dim in a.shape]
    else:
        new_shape = []
        for i, dim in enumerate(a.shape):
            dim += n if i == axis else 0
            new_shape.append(dim)

    out = np.zeros(new_shape, dtype=a.dtype)

    indexer = [slice(0, dim) for dim in a.shape]
    out[indexer] = a

    return out
