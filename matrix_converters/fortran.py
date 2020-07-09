import numpy as np
import pandas as pd
from six import string_types

from .common import coerce_matrix


def from_fortran_rectangle(file, n_columns, zones=None, tall=False, reindex_rows=False, fill_value=None):
    """Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to NOT be square. Also works
    with square matrices.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based
    positional index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns
    present must be known, since the format does not self-specify.

    Args:
        file(basestring or File): The file to read.
        n_columns (int): The number of columns in the matrix.
        zones (int, pandas.Index or None, optional): Defaults to ``None``. An `Index` or `Iterable` will be interpreted
            as the zone labels for the matrix rows and columns; returning a `DataFrame` or `Series` (depending on
            `tall`). If an `int` is provided, the returned `ndarray` will be truncated to this 'number of zones'.
        tall (bool, optional): Defaults to ``False``. If ``True``, a 'tall' version of the matrix will be returned.
        reindex_rows (bool, optional): Defaults to ``False``. If ``True``, and zones is an `Index`, the returned
            `DataFrame` will be reindexed to fill-in any missing rows.
        fill_value (int or float, optional): Defaults to ``None``. The value to pass to ``pandas.reindex()``

    Returns:
        ndarray, DataFrame or Series: A matrix from a FORTRAN-friendly .bin file that is known to not be square.

    Raises:
        AssertionError: if the shape is not valid.
    """

    if isinstance(file, string_types):
        with open(file, 'rb') as reader:
            return _from_fortran_binary(reader, n_columns, zones, tall, reindex_rows, fill_value)
    return _from_fortran_binary(file, n_columns, zones, tall, reindex_rows, fill_value)


def _from_fortran_binary(reader, n_columns, zones, tall, reindex_rows, fill_value):
    """Lower level function"""
    n_columns = int(n_columns)

    matrix = np.fromfile(reader, dtype=np.float32)
    rows = len(matrix) // (n_columns + 1)
    assert len(matrix) == (rows * (n_columns + 1))

    matrix.shape = rows, n_columns + 1
    # Convert binary representation from float to int, then subtract 1 since FORTRAN uses 1-based positional indexing
    row_index = np.frombuffer(matrix[:, 0].tobytes(), dtype=np.int32) - 1
    matrix = matrix[:, 1:]

    if zones is None:
        if tall:
            matrix.shape = matrix.shape[0] * matrix.shape[1]
        return matrix

    if isinstance(zones, (int, np.int_)):
        matrix = matrix[: zones, :zones]

        if tall:
            matrix.shape = zones * zones
        return matrix

    nzones = len(zones)
    matrix = matrix[: nzones, : nzones]
    row_labels = zones.take(row_index[:nzones])
    matrix = pd.DataFrame(matrix, index=row_labels, columns=zones)

    if reindex_rows:
        matrix = matrix.reindex_axis(zones, axis=0, fill_value=fill_value)

    if tall:
        return matrix.stack()
    return matrix


def from_fortran_square(file, zones=None, tall=False):
    """Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to be square.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based
    positional index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns
    present must be known, since the format does not self-specify. This method can infer the shape if it is square.

    Args:
        file (basestring or File): The file to read.
        zones (Index, int or None, optional): Defaults to ``None``. An `Index` or `Iterable` will be interpreted as the
            zone labels for the matrix rows and columns; returning a `DataFrame` or `Series` (depending on `tall`). If
            an `int` is provided, the returned `ndarray` will be truncated to this 'number of zones'. Otherwise, the
            returned `ndarray` will be size to the maximum number of zone dimensioned by the `Emmebank`.
        tall (bool, optional): Defaults to ``False``. If ``True``, a 1D data structure will be returned. If `zone_index`
            is provided, a Series will be returned, otherwise a 1D ndarray.

    Returns:
        DataFrame or ndarray: A matrix from a FORTRAN-friendly .bin file that is known to be square.
    """

    if isinstance(file, string_types):
        with open(file, 'rb') as reader:
            return _from_fortran_square(reader, zones, tall)
    return _from_fortran_square(file, zones, tall)


def _from_fortran_square(reader, zones, tall):
    """Lower level function"""
    floats = np.fromfile(reader, dtype=np.float32)
    n_words = len(floats)
    matrix_size = _infer_zones(n_words)
    floats.shape = matrix_size, matrix_size + 1

    data = floats[:, 1:]

    if zones is None:
        if tall:
            n = np.prod(data.shape)
            data.shape = n
            return data
        return data

    if isinstance(zones, (int, np.int_)):
        data = data[:zones, :zones]

        if tall:
            data.shape = zones * zones
            return data
        return data
    elif zones is None:
        return data

    zones = pd.Index(zones)
    n = len(zones)
    data = data[:n, :n]

    matrix = pd.DataFrame(data, index=zones, columns=zones)

    return matrix.stack() if tall else matrix


def _infer_zones(n_words):
    """Returns the inverse of n_words = matrix_size * (matrix_size + 1)"""
    n = int(0.5 + ((1 + 4 * n_words)**0.5)/2) - 1
    assert n_words == (n * (n + 1)), "Could not infer a square matrix from file"
    return n


def to_fortran(matrix, file, force_square=True, min_index=1):
    """Writes a FORTRAN-friendly .bin file (a.k.a. 'simple binary format'), in a square format.

    Args:
        matrix (DataFrame, Series or ndarray): The matrix to write to disk. If a `Series` is given, it MUST have a
            `MultiIndex` with exactly 2 levels to unstack.
        file (basestring or File): The path or file handler to write to.
        force_square (bool, optional): Defaults to ``True``.
        min_index (int): Defaults to ``1``.

    """
    array = coerce_matrix(matrix, force_square=force_square)

    if isinstance(file, string_types):
        with open(file, 'wb') as writer:
            _to_fortran(array, writer, min_index)
    else:
        _to_fortran(array, file, min_index)


def _to_fortran(array, writer, mindex):
    """Lower-level function"""
    rows, cols = array.shape
    temp = np.zeros([rows, cols + 1], dtype=np.float32)
    temp[:, 1:] = array

    index = np.arange(mindex, rows + mindex, dtype=np.int32)
    # Mask the integer binary representation as floating point
    index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32)
    temp[:, 0] = index_as_float

    temp.tofile(writer)
