*****
Mac
*****

1. Set Up a Virtual Environment
-------------------------------

To install ``shimming-toolbox``, we recommend that you use a virtual environment. Virtual environments are a tool to separate the Python environment and packages used between Python projects. They allow for different versions of Python packages to be installed and managed for the specific needs of your projects. There are several virtual environment managers available,
but the one we recommend and will use in our installation guide is
`conda <https://conda.io/docs/>`__, which is installed by default with Miniconda. We strongly recommend you create a virtual environment before you continue with your installation.

The Miniconda installation instructions depend on whether your system
default shell is Bash or Zsh. You can determine this from the output of
running the following in your terminal:

.. code:: bash

   echo $SHELL

Bash
~~~~

.. code:: bash

   cd
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda
   echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bash_profile
   source ~/.bash_profile

Zsh
~~~

.. code:: zsh

   cd
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda
   source $HOME/miniconda/bin/activate
   conda init zsh

Next, create your virtual environment:

.. code:: bash

   conda create -n shim_venv python=3.7

Then, activate your virtual environment:

.. code:: bash

   conda activate shim_venv

To switch back to your default environment, run:

.. code:: bash

   conda deactivate


2. Install dcm2niix
-------------------

Ensure that you have `dcm2niix <https://github.com/rordenlab/dcm2niix>`__ >= v1.0.20201102. installed on your system.

3. Install FSL
--------------

You will also need to install `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__.


4. Install from GitHub
----------------------

To install shimming-toolbox, clone
shimming-toolbox's repository (you will need to have Git installed on
your system):

.. code:: bash

  git clone https://github.com/shimming-toolbox/shimming-toolbox.git


Next, install using pip:

.. code:: bash

  cd shimming-toolbox
  pip install -e ".[docs,dev]"


5. Test the Install (optional)
------------------------------

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
