
CLI Reference
=============

The following section outlines the CLI of shimming-toolbox.

.. contents::
   :local:
   :depth: 2
..

Fieldmapping
------------

.. click:: shimmingtoolbox.cli.prepare_fieldmap:prepare_fieldmap_cli
   :prog: st_prepare_fieldmap

Shimming
--------


.. click:: shimmingtoolbox.cli.b0shim:b0shim_cli
   :prog: st_b0shim
   :nested: full

Masking
-------

.. click:: shimmingtoolbox.cli.mask:mask_cli
   :prog: st_mask
   :nested: full

File Conversion
---------------

.. click:: shimmingtoolbox.cli.dicom_to_nifti:dicom_to_nifti_cli
   :prog: st_dicom_to_nifti

Image manipulation
------------------

.. click:: shimmingtoolbox.cli.image:image_cli
   :prog: st_image
   :nested: full

.. click:: shimmingtoolbox.cli.maths:maths_cli
   :prog: st_maths
   :nested: full

Miscellaneous
-------------

.. click:: shimmingtoolbox.cli.download_data:download_data
   :prog: st_download_data

System Tools
------------

.. click:: shimmingtoolbox.cli.check_env:check_dependencies
   :prog: st_check_dependencies

.. click:: shimmingtoolbox.cli.check_env:dump_env_info
   :prog: st_dump_env_info
