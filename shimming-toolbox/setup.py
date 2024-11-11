from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(path.dirname(here), "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

path_version = path.join(here, 'shimmingtoolbox', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()

setup(
    name="shimmingtoolbox",
    python_requires=">=3.7",
    version=version,
    description="Code for performing real-time shimming using external MRI shim coils",
    long_description=long_description,
    url="https://github.com/shimming-toolbox/shimming-toolbox",
    author="NeuroPoly Lab, Polytechnique Montreal",
    author_email="neuropoly@googlegroups.com",
    keywords="",
    entry_points={
        'console_scripts': [
            "st_download_data=shimmingtoolbox.cli.download_data:download_data",
            "st_realtime_shim=shimmingtoolbox.cli.realtime_shim:realtime_shim_cli",
            "st_mask=shimmingtoolbox.cli.mask:mask_cli",
            "st_dicom_to_nifti=shimmingtoolbox.cli.dicom_to_nifti:dicom_to_nifti_cli",
            "st_prepare_fieldmap=shimmingtoolbox.cli.prepare_fieldmap:prepare_fieldmap_cli",
            "st_b1shim=shimmingtoolbox.cli.b1shim:b1shim_cli",
            "st_check_dependencies=shimmingtoolbox.cli.check_env:check_dependencies",
            "st_dump_env_info=shimmingtoolbox.cli.check_env:dump_env_info",
            "st_image=shimmingtoolbox.cli.image:image_cli",
            "st_maths=shimmingtoolbox.cli.maths:maths_cli",
            "st_b0shim=shimmingtoolbox.cli.b0shim:b0shim_cli",
            "st_create_coil_profiles=shimmingtoolbox.cli.create_coil_profiles:coil_profiles_cli",
            "st_sort_dicoms=shimmingtoolbox.cli.sort_dicoms:sort_dicoms",
            "st_unwrap=shimmingtoolbox.cli.unwrap:unwrap_cli"
        ]
    },
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
    install_requires=[
        "click",
        "dcm2bids>=3.0.1",
        'importlib-metadata ~= 4.0 ; python_version < "3.8"',
        "numpy>=1.21",
        "phantominator~=0.6.4",
        "nibabel>=3.2.1",
        "requests",
        "scipy>=1.7",
        "tqdm",
        "matplotlib>=3.5",
        "psutil>=5.8.0",
        "pydicom",
        "pytest>=6.2.5",
        "pytest-cov>=2.5.1",
        "scikit-learn>=1.1.2",
        "pillow>=9.0.0",
        "dataclasses",
        "raven",
        "joblib",
        "quadprog",
        "cloup",
    ],
    extras_require={
        'docs': ["sphinx>=1.7", "sphinx_rtd_theme==2.0.0", "sphinx-click", "myst_parser"],
        'dev': ["pre-commit>=2.10.0"]
    },
)
