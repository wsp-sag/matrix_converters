from setuptools import setup, find_packages

setup(
    name="matrix_converters",
    description="Converters for various matrix file formats used by travel demand forecasting models",
    author="WSP",
    maintainer="Brian Cheung",
    maintainer_email="brian.cheung@wsp.com",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "six"
    ],
    platforms="any"
)
