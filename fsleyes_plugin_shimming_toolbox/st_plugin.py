#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shimming Toolbox FSLeyes Plugin

This is an FSLeyes plugin that integrates the following ``shimmingtoolbox`` tools into FSLeyes' GUI:

- st_dicom_to_nifti
- st_mask
- st_prepare_fieldmap
- st_b0shim
- st_b1shim

---------------------------------------------------------------------------------------
Copyright (c) 2021 Polytechnique Montreal <www.neuro.polymtl.ca>
Authors: Alexandre D'Astous, Ainsleigh Hill, Gaspard Cereza, Julien Cohen-Adad
"""

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.views.canvaspanel as canvaspanel
import logging
import wx

from fsleyes_plugin_shimming_toolbox.tabs.b0shim_tab import B0ShimTab
from fsleyes_plugin_shimming_toolbox.tabs.b1shim_tab import B1ShimTab
from fsleyes_plugin_shimming_toolbox.tabs.dicom_to_nifti_tab import DicomToNiftiTab
from fsleyes_plugin_shimming_toolbox.tabs.fieldmap_tab import FieldMapTab
from fsleyes_plugin_shimming_toolbox.tabs.mask_tab import MaskTab


# We need to create a ctrlpanel.ControlPanel instance so that it can be recognized as a plugin by FSLeyes
# Class hierarchy: wx.Panel > fslpanel.FSLeyesPanel > ctrlpanel.ControlPanel
class STControlPanel(ctrlpanel.ControlPanel):
    """Class for Shimming Toolbox Control Panel"""

    # The CanvasPanel view is used for most FSLeyes plugins
    @staticmethod
    def supportedViews():
        return [canvaspanel.CanvasPanel]

    @staticmethod
    def defaultLayout():
        """This method makes the control panel appear on the top of the FSLeyes window."""
        return {"location": wx.TOP, "title": "Shimming Toolbox"}

    def __init__(self, parent, overlayList, displayCtx, ctrlPanel):
        """Initialize the control panel.

        Generates the widgets and adds them to the panel.

        """
        super().__init__(parent, overlayList, displayCtx, ctrlPanel)
        # Create a notebook with a terminal to navigate between the different functions.
        nb = NotebookTerminal(self)

        # Create the different tabs. Use 'select' to choose the default tab displayed at startup
        tab1 = DicomToNiftiTab(nb)
        tab2 = FieldMapTab(nb)
        tab3 = MaskTab(nb)
        tab4 = B0ShimTab(nb)
        tab5 = B1ShimTab(nb)
        nb.AddPage(tab1, tab1.title)
        nb.AddPage(tab2, tab2.title)
        nb.AddPage(tab3, tab3.title)
        nb.AddPage(tab4, tab4.title, select=True)
        nb.AddPage(tab5, tab5.title)

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(nb, 2, wx.EXPAND)
        self.sizer.Add(nb.terminal_component.sizer, 1, wx.EXPAND)
        self.sizer.AddSpacer(5)
        self.sizer.SetMinSize((600, 400))
        self.SetSizer(self.sizer)


class NotebookTerminal(wx.Notebook):
    """Notebook class with an extra terminal attribute"""
    def __init__(self, parent):
        super().__init__(parent)
        self.terminal_component = Terminal(parent)


class Terminal:
    """Create the terminal where messages are logged to the user."""
    def __init__(self, panel):
        self.panel = panel
        self.terminal = wx.TextCtrl(self.panel, wx.ID_ANY, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.terminal.SetDefaultStyle(wx.TextAttr(wx.WHITE, wx.BLACK))
        self.terminal.SetBackgroundColour(wx.BLACK)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.AddSpacer(5)
        self.sizer.Add(self.terminal, 1, wx.EXPAND)
        self.sizer.AddSpacer(5)

    def log_to_terminal(self, msg, level=None):
        if level is None:
            self.terminal.AppendText(f"{msg}\n")
        else:
            self.terminal.AppendText(f"{level}: {msg}\n")
