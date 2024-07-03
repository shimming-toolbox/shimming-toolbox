#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pathlib
import tempfile
import time
import wx

from .test_tabs import get_notebook, set_notebook_page, get_tab, get_all_children
from .. import realYield, run_with_orthopanel
from fsleyes_plugin_shimming_toolbox import __dir_testing__
from fsleyes_plugin_shimming_toolbox.tabs.dicom_to_nifti_tab import DicomToNiftiTab


def test_st_plugin_dcm2niix_run():
    run_with_orthopanel(_test_st_plugin_dcm2niix_run)


def _test_st_plugin_dcm2niix_run(view, overlayList, displayCtx):
    """ Makes sure dicom to nifti tab can be run (Add dummy input and simulate a click) """

    nb_terminal = get_notebook(view)

    # Select the dcm2niix tab
    assert set_notebook_page(nb_terminal, 'Dicom to Nifti')
    # Get the ST tab
    dcm2nifti_tab = get_tab(nb_terminal, DicomToNiftiTab)
    assert dcm2nifti_tab is not None

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        path_input = os.path.join(__dir_testing__, 'dicom_unsorted')

        # Fill in dicom2nifti tab options
        list_widgets = []
        get_all_children(dcm2nifti_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.TextCtrl) and widget.IsShown():
                if widget.GetName() == 'input':
                    widget.SetValue(path_input)
                    realYield()
                if widget.GetName() == 'subject':
                    widget.SetValue('test')
                    realYield()
                if widget.GetName() == 'output':
                    widget.SetValue(tmp)
                    realYield()

        # Call the function ran when clicking run button
        dcm2nifti_tab.run_component.run()

        # Search for files in the overlay for a maximum of 20s
        time_limit = 20  # s
        for i in range(time_limit):
            realYield()
            overlay_file = overlayList.find("sub-test_phase1")
            time.sleep(1)
            if overlay_file:
                break

        # Make sure there is an output in the overlay (that would mean the ST CLI ran)
        assert overlay_file is not None
