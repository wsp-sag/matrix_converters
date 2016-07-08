from setuptools import setup, find_packages

setup(name="matrix_converters",
      description="Converters for various matrix file formats used by travel demand forecasting models.",
      version="1.0",
      author="Peter Kucirek",
      author_email="pkucirek@pbworld.com",
      packages=find_packages(),
      requires=[ "numpy", "pandas"],
      platforms="any"
      )
