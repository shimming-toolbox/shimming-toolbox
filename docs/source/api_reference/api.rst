
API Reference
=============

The following section outlines the API of shimming-toolbox.

.. contents::
   :local:
   :depth: 1
..

Field mapping
-------------

.. automodule:: shimmingtoolbox.prepare_fieldmap
    :members:

.. automodule:: shimmingtoolbox.unwrap.unwrap_phase
   :members:

.. automodule:: shimmingtoolbox.unwrap.prelude
   :members:

.. automodule:: shimmingtoolbox.unwrap.skimage_unwrap
   :members:

Masking
-------

.. automodule:: shimmingtoolbox.masking.shapes
    :members:

.. automodule:: shimmingtoolbox.masking.threshold
    :members:

.. automodule:: shimmingtoolbox.masking.mask_utils
    :members:

.. automodule:: shimmingtoolbox.masking.mask_mrs
    :members:

.. automodule:: shimmingtoolbox.masking.softmasks
    :members:

Coils
-----

.. automodule:: shimmingtoolbox.coils.coil
    :members:
    :special-members: __init__

.. automodule:: shimmingtoolbox.coils.spherical_harmonics
    :members:

.. automodule:: shimmingtoolbox.coils.spher_harm_basis
    :members:

.. automodule:: shimmingtoolbox.coils.coordinates
    :members:

.. automodule:: shimmingtoolbox.coils.biot_savart
   :members:

Shim
----

Sequencer
_________
.. automodule:: shimmingtoolbox.shim.sequencer
   :members:

Shim Utils
__________
.. automodule:: shimmingtoolbox.shim.shim_utils
   :members:

B1 Shim
_______
.. automodule:: shimmingtoolbox.shim.b1shim
   :members:


Optimizer
---------

.. automodule:: shimmingtoolbox.optimizer.basic_optimizer
   :members:
   :special-members: __init__

.. automodule:: shimmingtoolbox.optimizer.optimizer_utils
   :members:
   :special-members: __init__
   :show-inheritance:
   :inherited-members:

.. automodule:: shimmingtoolbox.optimizer.lsq_optimizer
   :members:
   :special-members: __init__
   :show-inheritance:
   :inherited-members:

.. automodule:: shimmingtoolbox.optimizer.quadprog_optimizer
   :members:
   :special-members: __init__
   :show-inheritance:
   :inherited-members:

.. automodule:: shimmingtoolbox.optimizer.bfgs_optimizer
   :members:
   :special-members: __init__
   :show-inheritance:
   :inherited-members:

Nifti file handling
-------------------

.. automodule:: shimmingtoolbox.files.NiftiFile
   :members:
   :special-members: __init__
   :show-inheritance:

.. automodule:: shimmingtoolbox.files.NiftiFieldMap
   :members:
   :special-members: __init__
   :inherited-members:
   :show-inheritance:

.. automodule:: shimmingtoolbox.files.NiftiMask
   :members:
   :special-members: __init__
   :inherited-members:
   :show-inheritance:

.. automodule:: shimmingtoolbox.files.NiftiTarget
   :members:
   :special-members: __init__
   :inherited-members:
   :show-inheritance:

Image manipulation
------------------

.. automodule:: shimmingtoolbox.image
   :members:

Numerical model
---------------

.. automodule:: shimmingtoolbox.simulate.numerical_model
   :members:


Miscellaneous
-------------

Dicom to Nifti
______________
.. automodule:: shimmingtoolbox.dicom_to_nifti
   :members:

Load Nifti
__________
.. automodule:: shimmingtoolbox.load_nifti
   :members:

Download
________
.. automodule:: shimmingtoolbox.download
   :members:

PMU
________
.. automodule:: shimmingtoolbox.pmu
   :members:

Shimming toolbox utils
______________________
.. automodule:: shimmingtoolbox.utils
   :members:
