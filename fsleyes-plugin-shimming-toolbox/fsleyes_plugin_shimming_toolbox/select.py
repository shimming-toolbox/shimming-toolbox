#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__

logger = logging.getLogger(__name__)


def select_folder(event, tab, ctrl, focus=False):
    """Select a file folder from system path."""
    if focus:
        # Skip allows to handle other events
        focused = wx.Window.FindFocus()
        if ctrl != focused:
            if focused == tab:
                tab.terminal_component.log_to_terminal("Select a text box from the same row.", level="INFO")
                # If its the tab, don't handle the other events so that the message is only logged once
                return
            event.Skip()
            return

    dlg = wx.DirDialog(None, "Choose Directory", __CURR_DIR__, wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() == wx.ID_OK:
        folder = dlg.GetPath()
        ctrl.SetValue(folder)
        logger.info(f"Folder set to: {folder}")

    # Skip allows to handle other events
    event.Skip()


def select_file(event, tab, ctrl, focus=False):
    """Select a file from system path."""
    if focus:
        # Skip allows to handle other events
        focused = wx.Window.FindFocus()
        if ctrl != focused:
            if focused == tab:
                tab.terminal_component.log_to_terminal("Select a text box from the same row.", level="INFO")
                # If it's the tab, don't handle the other events so that the log message is only displayed once
                return
            event.Skip()
            return

    dlg = wx.FileDialog(parent=None,
                        message="Select File",
                        defaultDir=__CURR_DIR__,
                        style=wx.FD_DEFAULT_STYLE | wx.FD_FILE_MUST_EXIST)

    if dlg.ShowModal() == wx.ID_OK:
        fname = dlg.GetPath()
        ctrl.SetValue(fname)
        logger.info(f"File set to: {fname}")

    # Skip allows to handle other events
    event.Skip()


def select_from_overlay(event, tab, ctrl, focus=False):
    """Fetch path to file highlighted in the Overlay list.

    Args:
        event (wx.Event): event passed to a callback or member function.
        tab (Tab): Must be a subclass of the Tab class
        ctrl (wx.TextCtrl): the text item.
        focus (bool): Tells whether the ctrl must be in focus.
    """
    if focus:
        # Skip allows to handle other events
        focused = wx.Window.FindFocus()
        if ctrl != focused:
            if focused == tab:
                tab.terminal_component.log_to_terminal(
                    "Select a text box from the same row.",
                    level="INFO"
                )
                # If its the tab, don't handle the other events so that the log message is only once
                return
            event.Skip()
            return

    # This is messy and wont work if we change any class hierarchy.. using GetTopLevelParent() only
    # works if the pane is not floating
    # Get the displayCtx class initialized in STControlPanel
    window = tab.GetGrandParent()
    selected_overlay = window.displayCtx.getSelectedOverlay()
    if selected_overlay is not None:
        filename_path = selected_overlay.dataSource
        ctrl.SetValue(filename_path)
    else:
        tab.terminal_component.log_to_terminal(
            "Import and select an image from the Overlay list",
            level="INFO"
        )

    # Skip allows to handle other events
    event.Skip()
