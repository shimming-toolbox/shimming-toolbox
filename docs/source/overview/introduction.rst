.. _introduction:

Introduction
============

Context
_______

MRIs rely on magnetic fields to produce signals and make images. The main
magnetic field (B0) and the RF field (B1) need to be as homogeneous as
possible to ensure good image quality. If not, this can lead to reduced
signal, reduced SNR and artifacts. The act of homogenizing those fields
is called shimming. Many techniques have been developed over the years
that push the limits of active shimming.

Goals of the Shimming-Toolbox
_____________________________

The shimming toolbox aims to facilitate shimming by providing a suite of
software solutions, such as realtime shimming (using custom shim coils)
and realtime z-shimming. The shimming toolbox can process B0scans and
decompose them into a SH or non-orthogonal (custom) basis, allowing
researchers to extract the necessary shim parameters.
