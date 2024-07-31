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
import textwrap
import wx

from fsleyes_plugin_shimming_toolbox.tabs.b0shim_tab import B0ShimTab
from fsleyes_plugin_shimming_toolbox.tabs.b1shim_tab import B1ShimTab
from fsleyes_plugin_shimming_toolbox.tabs.dicom_to_nifti_tab import DicomToNiftiTab
from fsleyes_plugin_shimming_toolbox.tabs.fieldmap_tab import FieldMapTab
from fsleyes_plugin_shimming_toolbox.tabs.mask_tab import MaskTab

STLayout = textwrap.dedent(
    """
    fsleyes.views.orthopanel.OrthoPanel
    layout2|name=OrthoPanel 1;caption=Ortho View 1;state=67376064;dir=5;layer=0;row=0;pos=0;prop=100000;bestw=-1;besth=-1;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1;notebookid=-1;transparent=255|dock_size(5,0,0)=22|
    fsleyes.controls.orthotoolbar.OrthoToolBar,fsleyes.controls.overlaydisplaytoolbar.OverlayDisplayToolBar,fsleyes.controls.overlaylistpanel.OverlayListPanel,fsleyes.controls.locationpanel.LocationPanel,fsleyes_plugin_shimming_toolbox.st_plugin.STControlPanel;syncLocation=True,syncOverlayOrder=True,syncOverlayDisplay=True,syncOverlayVolume=True,movieRate=400,movieAxis=3;showCursor=True,bgColour=#000000ff,fgColour=#ffffffff,cursorColour=#00ff00ff,cursorGap=False,showColourBar=False,colourBarLocation=top,colourBarLabelSide=top-left,showXCanvas=True,showYCanvas=True,showZCanvas=True,showLabels=True,labelSize=12,layout=horizontal,xzoom=847.0127756479341,yzoom=847.0127756479341,zzoom=100.0
    layout2|name=Panel;caption=;state=768;dir=5;layer=0;row=0;pos=0;prop=100000;bestw=-1;besth=-1;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1;notebookid=-1;transparent=255|name=OrthoToolBar;caption=Ortho view toolbar;state=67382012;dir=1;layer=10;row=0;pos=0;prop=100000;bestw=572;besth=35;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1;notebookid=-1;transparent=255|name=OverlayDisplayToolBar;caption=Display toolbar;state=67382012;dir=1;layer=11;row=0;pos=0;prop=100000;bestw=953;besth=56;minw=-1;minh=-1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=-1;floath=-1;notebookid=-1;transparent=255|name=OverlayListPanel;caption=Overlay list;state=67373052;dir=3;layer=0;row=0;pos=0;prop=100000;bestw=197;besth=80;minw=1;minh=1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=197;floath=99;notebookid=-1;transparent=255|name=LocationPanel;caption=Location;state=67373052;dir=3;layer=0;row=0;pos=1;prop=100000;bestw=391;besth=111;minw=1;minh=1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=391;floath=130;notebookid=-1;transparent=255|name=STControlPanel;caption=Shimming Toolbox;state=67373052;dir=1;layer=0;row=0;pos=0;prop=100000;bestw=600;besth=400;minw=1;minh=1;maxw=-1;maxh=-1;floatx=-1;floaty=-1;floatw=600;floath=419;notebookid=-1;transparent=255|dock_size(5,0,0)=22|dock_size(3,0,0)=176|dock_size(1,10,0)=37|dock_size(1,11,0)=58|dock_size(1,0,0)=384|
    """
)


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
        self.sizer.AddSpacer(5)
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
