#!/usr/bin/python3
# -*- coding: utf-8 -*-

import wx

from .. import realYield, run_with_orthopanel
from fsleyes_plugin_shimming_toolbox.st_plugin import STControlPanel, NotebookTerminal


def test_st_plugin_loads():
    run_with_orthopanel(_test_st_plugin_loads)


def _test_st_plugin_loads(view, overlayList, displayCtx):
    view.togglePanel(STControlPanel)
    realYield()


def test_st_plugin_tabs_exist():
    run_with_orthopanel(_test_st_plugin_tabs_exist)


def _test_st_plugin_tabs_exist(view, overlayList, displayCtx):
    nb_terminal = get_notebook(view)

    tabs = nb_terminal.GetChildren()
    assert len(tabs) > 0


def get_notebook(view):
    """ Returns the notebook terminal from the ST plugin."""

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


def set_notebook_page(nb_terminal, page_name):
    """ Sets the notebook terminal to the page with the given name."""
    for i in range(nb_terminal.GetPageCount()):
        if nb_terminal.GetPageText(i) == page_name:
            nb_terminal.SetSelection(i)
            realYield()
            return True
    return False


def get_tab(nb_terminal, tab_instance):
    """ Returns the tab instance from the ST notebook."""
    tabs = nb_terminal.GetChildren()
    for tab in tabs:
        if isinstance(tab, tab_instance):
            return tab


def get_all_children(item, list_widgets, depth=None):
    """ Returns all children and sub children from an item."""
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


def set_dropdown_selection(dropdown_widget, selection_name):
    """ Sets the notebook terminal to the page with the given name."""
    for i in range(dropdown_widget.GetCount()):
        if dropdown_widget.GetString(i) == selection_name:
            dropdown_widget.SetSelection(i)
            wx.PostEvent(dropdown_widget.GetEventHandler(), wx.CommandEvent(wx.EVT_CHOICE.typeId, dropdown_widget.GetId()))
            realYield()
            return True
    return False
