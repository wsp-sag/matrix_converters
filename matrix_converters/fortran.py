import numpy as np
import pandas as pd

from common import coerce_matrix


def from_binary_matrix(file):
    """
    Reads a .bin file compatible with Bill Davidson's FORTRAN code

    Args:
        file (basestring or File): The file to read.

    Returns:
        Numpy.ndarray of matrix data

    """
    if isinstance(file, basestring):
        with open(file, 'rb') as reader:
            return _from_binary_matrix(reader)
    return _from_binary_matrix(file)


def _from_binary_matrix(reader):
    """Lower level function"""
    floats = np.fromfile(reader, dtype=np.float32)
    n_words = len(floats)
    matrix_size = _infer_zones(n_words)
    floats.shape = matrix_size, matrix_size + 1

    return floats[:, 1:]


def _infer_zones(n_words):
    """Returns the inverse of n_words = matrix_size * (matrix_size + 1)"""
    return int(0.5 + ((1 + 4 * n_words)**0.5)/2) - 1


def to_binary_matrix(matrix, file):
    """
    Writes a matrix to .bin file compatible with Bill Davidson's FORTRAN code

    Args:
        matrix (DataFrame or Series or ndarray): The matrix to write to disk. If a Series is given, it MUST have a
            MultiIndex with exactly 2 levels to unstack.
        file (basestring or File): The path or file handler to write to.

    """
    array = coerce_matrix(matrix)

    if isinstance(file, basestring):
        with open(file, 'wb') as writer:
            _to_binary_matrix(array, writer)
    else:
        _to_binary_matrix(array, file)


def _to_binary_matrix(array, writer):
    """Lower-level function"""
    n = array.shape[0]
    temp = np.zeros([n, n + 1], dtype=np.float32)
    temp[:, 1:] = array

    index = np.arange(1, n+1, dtype=np.int32)
    # Mask the integer binary representation as floating point
    index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32)
    temp[:, 0] = index_as_float

    temp.tofile(writer)

