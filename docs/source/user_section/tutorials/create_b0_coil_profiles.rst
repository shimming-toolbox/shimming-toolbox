.. _create_b0_coil_profiles:

Create B0 Coil Profiles
-----------------------

This tutorial describes how to create custom B0 coil profiles to be used by ``shimming-toolbox``. Coil profiles can be
calculated by acquiring field maps at different currents for each of the different coil channels. For each channel, a
linear regression between the field maps and the current yields the coil profiles.

In this tutorial, we will be characterising an 8 channel coil that was acquired with a dual echo field mapping sequence at 2
different currents for each channel (-0.5 amps and 0.5 amps). We will start by processing the DICOMs into NIfTI files, we will then fill the
configuration file containing the necessary information to correctly and accurately process the coil profiles. The last
step consists in running :ref:`st_create_coil_profiles` on the command line to calculate the coil profiles.

.. Note::

    Prelude is necessary to be able to calculate the field maps. To download and install Prelude, see `FSL's install page <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__.

Download data
_____________

Open a Terminal and download this dataset by running the following command:

.. code:: bash

    st_download_data data_create_coil_profiles

Go to the dataset folder:

.. code:: bash

    cd data_create_coil_profiles

The dataset contains field mapping acquisitions in DICOM format to characterize an 8 channel coil.

DICOM to NIFTI
______________

The first step is to convert the DICOMs into NIfTI files. To do so, we first need to sort them by their acquisition names.

.. code:: bash

    st_sort_dicoms -i ./dicoms -o ./dicoms_sorted

The command :ref:`st_create_coil_profiles` requires the different phase, magnitude, echoes and currents to be separated
in different NIfTI files. A helper script was downloaded with the dataset named: `batch_dicom_to_nifti.sh` that will
process each folder of DICOMs into NIfTI files sorted by the folder names of the input. This script can be used with other datasets.
You will need to give executable permission to the script beforehand.

.. code:: bash

    chmod +x batch_dicom_to_nifti.sh
    ./batch_dicom_to_nifti.sh dicoms_sorted .

Config file
___________

The configuration file allows :ref:`st_create_coil_profiles` to know the number of channels, the path to the different
NIfTI folders, the current used for each channel and other information that will allow to generate the config file
required for B0 shimming (:ref:`st_b0shimming`). The configuration file for this dataset is already filled in as:
`configuration_file.json`. The following describes the different arguments required in the JSON file:

* "phase": 3D list containing the path of phase NIfTI files of the different channels, currents and echos. Note that the first dimension is the different channels, the second the different currents and the third the different echoes.

* "mag": 3D list containing the path of magnitude NIfTI files of the different channels, currents and echoes.

* "setup_currents": 2D list containing the currents used for each channel. Note that the first dimension is the different channels and the second is the different currents.

* "name": Name of the coil

* "n_channels": Number of channels

* "Units": Units used for setup_currents. Note that this is for displayed text purposes and does not affect any coil profile output.

* "coef_channel_minmax": 2D list containing the minimum and maximum currents allowed for each channel when shimming. Note that the first dimension is for the channels and the second dimension for the minimum and maximum current.

* "coef_sum_max": Maximum total current that the coil can use during shimming. Use null if there is not a limit on the total current.

Create the coil profiles
________________________

The following command will compute the coil profiles. In more details, a mask is computed using the magnitude of all
channels, currents and echoes. The 'threshold' option can be used to change the mask threshold. Fieldmaps are computed for
each current and channel. A linear regression is then performed for each channel to obtain the coil profiles.

.. code:: bash

    st_create_coil_profiles --input "demo_config_coil_profile.json" --unwrapper "prelude" --threshold 0.03 --output "coil_profiles.nii.gz" --relative-path .

The coil profiles are in a NIfTI file named "coil_profiles.nii.gz". To visualize them, launch FSLeyes with the following command:

.. code:: bash

    shimming-toolbox

and drag the file "coil_profiles.nii.gz" in the FSLeyes window. The coil profiles in this demo are in Hz/A.

To create your own custom coil
______________________________

When creating your own custom coil using the commands above, keep in mind the following:

* :ref:`st_create_coil_profiles` will automatically scale Siemens phase data to radians. For other vendors, a step to rescale phase data to [-pi, pi] is necessary before using the command :ref:`st_create_coil_profiles`.

* The output B0 coil profile is scaled in Hz/<current> where current depends on the value in the configuration file. For example, this tutorial could have use 500 mA instead of 0.5 A. This would have resulted in a coil profile in Hz/mA instead of Hz/A.
