#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pathlib
import tempfile
import time
import wx

from .. import realYield, run_with_orthopanel, simclick
from fsleyes_plugin_shimming_toolbox import __dir_testing__


def test_st_plugin_loads():
    run_with_orthopanel(_test_st_plugin_loads)


def _test_st_plugin_loads(view, overlayList, displayCtx):
    from fsleyes_plugin_shimming_toolbox.st_plugin import STControlPanel
    view.togglePanel(STControlPanel)
    realYield()


def test_st_plugin_tabs_exist():
    run_with_orthopanel(_test_st_plugin_tabs_exist)


def _test_st_plugin_tabs_exist(view, overlayList, displayCtx):
    nb_terminal = get_notebook(view)

    tabs = nb_terminal.GetChildren()
    assert len(tabs) > 0


def test_st_plugin_dcm2niix_run():
    run_with_orthopanel(_test_st_plugin_dcm2niix_run)


def _test_st_plugin_dcm2niix_run(view, overlayList, displayCtx):
    """ Makes sure dicom to nifti tab can be run (Add dummy input and simulate a click) """
    from fsleyes_plugin_shimming_toolbox.tabs.dicom_to_nifti_tab import DicomToNiftiTab
    nb_terminal = get_notebook(view)

    # Select the dcm2niix tab
    nb_terminal.SetSelection(0)
    realYield()
    tabs = nb_terminal.GetChildren()
    dcm2nifti_tab = None
    for tab in tabs:
        if isinstance(tab, DicomToNiftiTab):
            dcm2nifti_tab = tab
            break

    assert dcm2nifti_tab is not None

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        path_input = os.path.join(__dir_testing__, 'dicom_unsorted')

        # Fill in dicom2nifti tab options
        list_widgets = []
        get_all_children(dcm2nifti_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.TextCtrl):
                if widget.GetName() == 'input':
                    widget.SetValue(path_input)
                    realYield()
                if widget.GetName() == 'subject':
                    widget.SetValue('test')
                    realYield()
                if widget.GetName() == 'output':
                    widget.SetValue(tmp)
                    realYield()

        # Simulate a mouse click on the run button
        sim = wx.UIActionSimulator()
        for widget in list_widgets:
            if isinstance(widget, wx.Button):
                if widget.GetLabel() == 'Run':
                    simclick(sim, widget)

        # Search for files in the overllay for a maximum of 20s
        time_limit = 20  # s
        for i in range(time_limit):
            realYield()
            overlay_file = overlayList.find("sub-test_phase1")
            time.sleep(1)
            if overlay_file:
                break

        # Make sure there is an output in the overlay (that would mean the ST CLI ran)
        assert overlay_file is not None


def get_notebook(view):

    from fsleyes_plugin_shimming_toolbox.st_plugin import STControlPanel
    from fsleyes_plugin_shimming_toolbox.st_plugin import NotebookTerminal

    ctrl = view.togglePanel(STControlPanel)
    realYield()
    children = ctrl.sizer.GetChildren()

    nb_terminal = None
    for child in children:
        tmp = child.GetWindow()
        if isinstance(tmp, NotebookTerminal):
            nb_terminal = tmp
            break

    assert nb_terminal is not None
    return nb_terminal


def get_all_children(item, list_widgets, depth=None):
    if depth is not None:
        depth -= 1

    if depth is not None and depth < 0:
        return

    for sizerItem in item.GetChildren():
        widget = sizerItem.GetWindow()
        if not widget:
            # then it's probably a sizer
            sizer = sizerItem.GetSizer()
            if isinstance(sizer, wx.Sizer):
                get_all_children(sizer, list_widgets, depth=depth)
        else:
            list_widgets.append(widget)
