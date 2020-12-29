import os

# https://packaging.python.org/guides/single-sourcing-package-version/
try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata
__version__ = metadata.version(__name__)
del metadata

file_path = os.path.dirname(os.path.realpath(__file__))
__dir_shimmingtoolbox__ = os.path.dirname(file_path)
del file_path

__dir_testing__ = os.path.join(__dir_shimmingtoolbox__, 'testing_data')

config_path = __dir_shimmingtoolbox__, 'config', 'dcm2bids.json'
__dir_config_dcm2bids__ = os.path.join(config_path)
del config_path

from .dicom_to_nifti import dicom_to_nifti
