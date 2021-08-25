import os

# https://packaging.python.org/guides/single-sourcing-package-version/
try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata
__version__ = metadata.version(__name__)
del metadata

__dir_shimmingtoolbox__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__dir_testing__ = os.path.join(__dir_shimmingtoolbox__, 'testing_data')
__dir_config_dcm2bids__ = os.path.join(__dir_shimmingtoolbox__, 'config', 'dcm2bids.json')
__dir_config_scanner_constraints__ = os.path.join(__dir_shimmingtoolbox__, 'config', 'coil_constraints.json')

from .dicom_to_nifti import dicom_to_nifti
