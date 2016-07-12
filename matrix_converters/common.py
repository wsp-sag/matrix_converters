import numpy as np
import pandas as pd


def coerce_matrix(matrix, allow_raw=True):
    """
    Infers a NumPy array from given input

    Args:
        matrix (DataFrame or Series or ndarray or Iterable):

    Returns:
        2D ndarray of type float32
    """
    if isinstance(matrix, pd.DataFrame):
        assert matrix.index.equals(matrix.columns)
        return matrix.values.astype(np.float32)
    elif isinstance(matrix, pd.Series):
        assert matrix.index.nlevels == 2, "Cannot infer a matrix from a Series with more or fewer than 2 levels"
        wide = matrix.unstack()

        union = wide.index | wide.columns
        wide = wide.reindex_axis(union, fill_value=0.0, axis=0).reindex_axis(union, fill_value=0.0, axis=1)
        return wide.values.astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    assert len(matrix.shape) == 2
    i,j = matrix.shape
    assert i == j

    return matrix


def expand_array(a, n):
    """
    Expands an array across all dimensions by a set amount

    Args:
        a: The array to expand
        n: The (non-negative) number of items to expand by.

    Returns: The expanded array
    """

    new_shape = [i + n for i in a.shape]

    out = np.zeros(new_shape, dtype=a.dtype)

    indexer = [slice(0, i) for i in a.shape]
    out[indexer] = a

    return out
