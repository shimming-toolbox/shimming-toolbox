from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

# Read the long description from README.rst
with open(path.join(path.dirname(here), "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# Read the version from version.txt
path_version = path.join(here, 'shimmingtoolbox', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()

# Determine install_requires from requirements.txt if it exists
requirements_path = path.join(path.dirname(here), "requirements_st.txt")
if path.exists(requirements_path):
    with open(requirements_path) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    install_requires = [
        "click",
        "cloup",
        "dataclasses",
        "dcm2bids>=3.0.1",
        "dcm2niix>=1.0.20241211",
        "importlib-metadata",
        "joblib",
        "matplotlib>=3.5",
        "nibabel>=3.2.1",
        "numpy>=1.21",
        "phantominator~=0.6.4",
        "pillow>=9.0.0",
        "psutil>=5.8.0",
        "pydicom",
        "pytest>=6.2.5",
        "pytest-cov>=2.5.1",
        "quadprog",
        "raven",
        "requests",
        "scikit-learn>=1.1.2",
        "scipy>=1.7",
        "spec2nii",
        "tqdm"
    ]

setup(
    name="shimmingtoolbox",
    python_requires=">=3.10",
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
    install_requires=install_requires,
    extras_require={
        'docs': ["sphinx>=1.7", "sphinx_rtd_theme==2.0.0", "sphinx-click", "myst_parser"],
        'dev': ["pre-commit>=2.10.0"]
    },
)