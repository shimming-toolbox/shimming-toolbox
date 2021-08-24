*****
Linux
*****

Dependencies
------------


Install FSL
===========

You will need to install `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__ if you want to use ``prelude``.


Install
~~~~~~~

To install ``shimming-toolbox``, clone the ``shimming-toolbox`` repository from ``GitHub`` (you will need to have ``Git`` installed on your system):

.. code:: bash

  git clone https://github.com/shimming-toolbox/shimming-toolbox.git


Next, install ``shimming-toolbox`` using the ``Makefile``:

.. code:: bash

  cd shimming-toolbox
  make install


Test the Install (optional)
---------------------------

Comprehensive Test
~~~~~~~~~~~~~~~~~~

To run the entire testing suite, run ``pytest`` from the
shimming-toolbox directory:

.. code:: bash

 cd shimming-toolbox
 pytest

See https://docs.pytest.org/ for more options.

If all tests pass, shimming-toolbox was installed successfully.

Testing subsets of soft dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prelude`` and ``dcm2niix`` are soft dependencies, so you may wish to run the
parts of the testing suite that do not depend on them.

To test shimming-toolbox without ``prelude`` and without ``dcm2niix``:

.. code:: bash

 cd shimming-toolbox
 pytest -m "not prelude and not dcm2niix"

To test shimming-toolbox without ``prelude`` and with ``dcm2niix``, you can use the above block but modifying the ``-m`` argument to ``"not prelude"``.

To test shimming-toolbox with ``prelude`` and without ``dcm2niix``, you can use the above block but modifying the ``-m`` argument to ``"not dcm2niix"``.

To test **only** the parts of shimming-toolbox dependent on ``prelude`` or
``dcm2niix``, the corresponding ``-m`` argument is ``"prelude or dcm2niix"``

Note that supplying the ``"-m"`` argument ``"prelude and dcm2niix"`` only runs tests dependent on both ``prelude`` **and** ``dcm2niix``.


For Developers
==============

The installation files can be found in the ``installer`` folder, and are called by the ``Makefile``.

When you run ``make install``, we first check if the ``ST_DIR`` exists, or if a clean install has
been requested. The ``ST_DIR`` is where this package and also the ``fsleyes-plugin-shimming-toolbox`` are installed. By choosing clean, you delete the entire install directory, and consequently any prior installs of ``shimming-toolbox`` or ``fsleyes-plugin-shimming-toolbox``. Note that this is set to ``CLEAN==false`` by default.

We next check if ``conda`` has been installed into the ``ST_DIR``. If not, we run the ``conda`` installer.

Finally, we create a virtual environment and install ``shimming-toolbox``.
