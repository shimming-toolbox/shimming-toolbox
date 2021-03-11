from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="shimmingtoolbox",
    python_requires=">=3.7",
    version="0.1.0",
    description="Code for performing real-time shimming using external MRI shim coils",
    long_description=long_description,
    url="https://github.com/shimming-toolbox/shimming-toolbox",
    author="NeuroPoly Lab, Polytechnique Montreal",
    author_email="neuropoly@googlegroups.com",
    keywords="",
    entry_points={
        'console_scripts': [
            "st_download_data=shimmingtoolbox.cli.download_data:download_data",
            "st_realtime_zshim=shimmingtoolbox.cli.realtime_zshim:realtime_zshim_cli",
            "st_mask=shimmingtoolbox.cli.mask:mask_cli",
            "st_dicom_to_nifti=shimmingtoolbox.cli.dicom_to_nifti:dicom_to_nifti_cli",
            "st_prepare_fieldmap=shimmingtoolbox.cli.prepare_fieldmap:prepare_fieldmap_cli",
            "st_check_dependencies=shimmingtoolbox.cli.check_env:check_dependencies",
            "st_dump_env_info=shimmingtoolbox.cli.check_env:dump_env_info"
        ]
    },
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "click",
        "dcm2bids==2.1.4",
        'importlib-metadata ~= 1.0 ; python_version < "3.8"',
        "numpy~=1.19.0",
        "phantominator~=0.6.4",
        "nibabel~=3.1.1",
        "requests",
        "scipy~=1.5.0",
        "tqdm",
        "matplotlib~=3.1.2",
        "psutil~=5.7.3",
        "pytest~=4.6.3",
        "pytest-cov~=2.5.1",
        "sklearn~=0.0",
        "pillow~=8.0",
        "dataclasses",
        "raven",
    ],
    extras_require={
        'docs': ["sphinx>=1.6", "sphinx_rtd_theme>=0.2.4", "sphinx-click"],
        'dev': ["pre-commit>=2.10.0"]
    },
)
