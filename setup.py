from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shimmingtoolbox",
    python_requires=">=3.6",
    version="0.1.0",
    description="Code for performing real-time shimming using external MRI shim coils",
    long_description=long_description,
    url="https://github.com/shimming-toolbox/shimming-toolbox-py",
    author="NeuroPoly Lab, Polytechnique Montreal",
    author_email="neuropoly@googlegroups.com",
    keywords="",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        #"numpy~=1.16.0",
        'importlib-metadata ~= 1.0 ; python_version < "3.8"',
    ],
    extras_require={
        'testing': ["pytest~=4.6.3", "pytest-cov~=2.5.1"]
    },
)
