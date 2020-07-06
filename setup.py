from setuptools import setup, find_packages
from os import path

import shimmingtoolbox

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

req_path = path.join(here, "requirements.txt")
with open(req_path, "r") as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setup(
    name="shimmingtoolbox",
    python_requires=">=3.7",
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
            "shim-b0maps=shimmingtoolbox.cli.b0maps:main"
        ]
    },
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=install_reqs,
    package_dir={"shimmingtoolbox": "shimmingtoolbox"},
)
