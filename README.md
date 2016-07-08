# Matrix Converters

This package bundles functions which read and write various matrix file formats using nothing but the Scientific Python
stack (NumPy and Pandas). The idea is to allow closed-source software (e.g. Emme, VISUM, Cube, etc.) to "talk" to each
other through matrices.

Each format comes with a reader and writer, which converts to and from NumPy arrays or Pandas DataFrames. Pandas Series
with 2-level MultiIndex are also supported.

## Currently Supported Matrix Formats

I hope this list to grow over time with other formats.

- Emme 'internal' matrices (inside the Emmebank)
- Emme 'official binary' matrices
- FORTRAN-friendly BIN files (to work with Bill Davidson's code)


## License

(c) WSP | Parsons Brinckerhoff

To be used by company staff only.