.. _introduction:

Introduction
============

Context
_______

Inhomogeneities of the main magnetic field (B0) can lead to image artefacts and
signal loss in MRI. Correcting for these inhomogeneities and producing a flat,
homogeneous field is called "shimming". When such shimming is done during the
acquisition of MRI images, it is referred to as "active shimming". Many
techniques have been developed for active shimming over the years, pushing the
limits of hardware and software.

Goals of the Shimming-Toolbox
_____________________________

We developed the Shimming Toolbox to facilitate the deployment of these
techniques, such as realtime shimming using custom coils, or
realtime z-shimming. Using the Toolbox, B0 maps can be calculated and decomposed
onto a spherical harmonic, or custom non-orthogonal basis, and the shim
parameters necessary to create a flat B0 field can be calculated.
