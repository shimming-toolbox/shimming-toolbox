Installation
============

We recommend that shimming-toolbox be used with
`Miniconda <https://conda.io/docs/glossary.html#miniconda-glossary>`__,
a lightweight version of the `Anaconda
distribution <https://www.anaconda.com/distribution/>`__. Miniconda is
typically used to create virtual Python environments, which provides a
separation of installation dependencies between different Python
projects. Although it is possible to install shimming-toolbox without
Miniconda or virtual environments, we only provide instructions for this
recommended installation setup.

First, verify that you have a compatible version of Miniconda or
Anaconda properly installed and in your system path.

In a new terminal window (macOS or Linux) or Anaconda Prompt (Windows –
if it is installed), run the following command:

.. code:: bash

   conda search python

If a list of available Python versions are displayed and versions
>=3.7.0 are available, you may skip to the next section (Git).

Linux
-----

To install Miniconda, run the following commands in your terminal:

.. code:: bash

   cd
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda
   echo ". ~/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
   source ~/.bashrc

macOS
-----

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

Windows
-------

.. NOTE ::
   The shimming-toolbox installation instruction using the
   Miniconda have only been tested for Windows 10. Older versions of
   Windows may not be compatible with the tools required to run
   shimming-toolbox.

To install Miniconda, go to the `Miniconda installation
website <https://conda.io/miniconda.html>`__ and click on the Python 3.x
version installer compatible with your Windows system (64-bit
recommended). After the download is complete, execute the downloaded
file, and follow the instructions. If you are unsure about any of the
installation options, we recommend you use the default settings.

Git (optional)
--------------

Git is a software version control system. Because shimming-toolbox is
hosted on GitHub, a service that hosts Git repositories, having Git
installed on your system allows you to download the most up-to-date
development version of shimming-toolbox from a terminal, and also
allows you to contribute to the project if you wish to do so.

Although an optional step (shimming-toolbox can also be downloaded
other ways, see below), if you want to install Git, please follow
instructions for your operating system on the `Git
website <https://git-scm.com/downloads>`__.

Virtual Environment
-------------------

Virtual environments are a tool to separate the Python environment and
packages used between Python projects. They allow for different versions
of Python packages to be installed and managed for the specific needs of
your projects. There are several virtual environment managers available,
but the one we recommend and will use in our installation guide is
`conda <https://conda.io/docs/>`__, which is installed by default with
Miniconda. We strongly recommend you create a virtual environment before
you continue with your installation.

Although the shimming-toolbox package on PyPI supports Python versions 3.7
and greater, we recommend you only use shimming-toolbox with Python 3.7, as
our tests only cover that version for now.

To create a Python 3.7 virtual environment named "shim_venv", in a
terminal window (macOS or Linux) or Anaconda Prompt (Windows) run the
following command and answer "y" to the installation instructions:

.. code:: bash

   conda create -n shim_venv python=3.7

Then, activate your virtual environment:

.. code:: bash

   conda activate shim_venv

To switch back to your default environment, run:

.. code:: bash

   conda deactivate

shimming-toolbox
----------------

Development version
~~~~~~~~~~~~~~~~~~~

Ensure that you have ``dcm2niix`` installed on your system.

You will also need to install `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__ using Python 2.

To install the development version of shimming-toolbox, clone
shimming-toolbox's repository (you will need to have Git installed on
your system):

.. code:: bash

   git clone https://github.com/shimming-toolbox/shimming-toolbox.git

If you don’t have Git installed, download and extract
shimming-toolbox from this
`link <https://github.com/shimming-toolbox/shimming-toolbox/archive/master.zip>`__.

Then, in your Terminal, go to the shimming-toolbox folder and install
the shimming-toolbox package. The following ``cd`` command assumes
that you followed the ``git clone`` instruction above:

.. code:: bash

   cd shimming-toolbox
   pip install -e ".[docs,dev]"

.. NOTE ::
   If you downloaded shimming-toolbox using the link above
   instead of ``git clone``, you may need to cd to a different folder
   (e.g. ``Downloads`` folder located within your home folder ``~``), and
   the shimming-toolbox folder may have a different name
   (e.g. ``shimming-toolbox-master``).

Updating
^^^^^^^^

To update an already cloned shimming-toolbox package, pull the latest
version of the project from GitHub and reinstall the application:

.. code:: bash

   cd shimming-toolbox
   git pull
   pip install -e ".[docs,dev]"

Testing the installation
------------------------

Comprehensive test
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
