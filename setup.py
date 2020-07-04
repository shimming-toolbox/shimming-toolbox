from setuptools import setup, find_packages
from os import path

import shimmingtoolbox

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shimmingtoolbox",
    python_requires=">=3.6",
    version=shimmingtoolbox.__version__,
    description="Code for performing real-time shimming using external MRI shim coils",
    long_description=long_description,
    url="https://github.com/shimming-toolbox/shimming-toolbox-py",
    author="NeuroPoly Lab, Polytechnique Montreal",
    author_email="neuropoly@googlegroups.com",
    keywords="",
    entry_points={
        'console_scripts': [
            "shim-referencemaps=shimmingtoolbox.cli.referencemaps:main",
            "shim-b0maps=shimmingtoolbox.cli.b0map:main"
        ]
    },
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "numpy~=1.19.0",
        "phantominator~=0.6.4",
    ],
    extras_require={
        'testing': ["pytest~=4.6.3", "pytest-cov~=2.5.1"]
    },
)
