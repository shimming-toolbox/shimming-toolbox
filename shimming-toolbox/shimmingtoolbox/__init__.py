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
__dir_repo__ = os.path.dirname(__dir_shimmingtoolbox__)
__dir_testing__ = os.path.join(__dir_shimmingtoolbox__, 'testing_data')
__config_dcm2bids__ = os.path.join(__dir_shimmingtoolbox__, 'shimmingtoolbox', 'config', 'dcm2bids.json')
__config_scanner_constraints__ = os.path.join(__dir_shimmingtoolbox__,
                                              'shimmingtoolbox',
                                              'config',
                                              'scanner_coil_constraints.json')
__config_custom_coil_constraints__ = os.path.join(__dir_shimmingtoolbox__,
                                                  'shimmingtoolbox',
                                                  'config',
                                                  'custom_coil_constraints.json')

from .dicom_to_nifti import dicom_to_nifti
