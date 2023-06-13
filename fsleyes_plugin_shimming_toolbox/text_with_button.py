#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __DIR_ST_PLUGIN_IMG__
from fsleyes_plugin_shimming_toolbox.select import select_file, select_folder, select_from_overlay

# Load icon resources
info_icon = wx.Image(os.path.join(__DIR_ST_PLUGIN_IMG__, 'info-icon.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap()
asterisk_icon = wx.Image(os.path.join(__DIR_ST_PLUGIN_IMG__, 'asterisk.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap()


class TextWithButton:
    """Creates a button with an input text box.

    wx.BoxSizer:

        InfoIcon(wx.StaticBitmap) - info icon
        wx.Button - clickable input button
        [wx.TextCtrl] - input text box(es)
        wx.StaticBitmap - asterisk icon

    Attributes:

        panel (wx.Panel): Instance of a Panel, this is usually a Tab
        button_label (str): label to be put on the button.
        button_function: function which gets called when the button is clicked on. If it's a list, assign to the
                         'n_text_boxes'
        default_text (str): (optional) default text to be displayed in the input text box.
        textctrl_list (list wx.TextCtrl): list of input text boxes, can be more than one in a row.
        n_text_boxes (int): number of input text boxes to create.
        name (str): name of the cli option
        info_text (str): text to be displayed when hovering over the info icon; should describe
            what the button/input is for.
        required (bool): if this input is required or not. If True, a red asterisk will be
            placed next to the input text box to indicate this.
    """

    def __init__(self, panel, button_label, button_function, name="default", default_text="",
                 n_text_boxes=1, info_text="", required=False, load_in_overlay=False):
        self.panel = panel
        self.button_label = button_label
        if type(button_function) is not list:
            button_function = [button_function]
        self.button_function = button_function
        self.default_text = default_text
        self.textctrl_list = []
        self.n_text_boxes = n_text_boxes
        self.name = name
        self.info_text = info_text
        self.required = required
        self.load_in_overlay = load_in_overlay

    def create(self):
        text_with_button_box = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self.panel, -1, label=self.button_label)
        text_with_button_box.Add(
            create_info_icon(self.panel, self.info_text), 0, wx.ALIGN_LEFT | wx.RIGHT, 7
        )
        for i_text_box in range(0, self.n_text_boxes):
            textctrl = wx.TextCtrl(parent=self.panel, value=self.default_text, name=self.name)
            self.textctrl_list.append(textctrl)

        if len(self.button_function) > self.n_text_boxes:
            raise RuntimeError("button_function has more functions than the number of input boxes")

        if len(self.button_function) == 1:
            focus = False
        else:
            focus = True

        for i, button_function in enumerate(self.button_function):
            if button_function == "select_folder":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i]: \
                    select_folder(event, panel, ctrl, focus)
                button.Bind(wx.EVT_BUTTON, function)
            elif button_function == "select_file":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i]: \
                    select_file(event, panel, ctrl, focus)
                button.Bind(wx.EVT_BUTTON, function)
            elif button_function == "select_from_overlay":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i]: \
                    select_from_overlay(event, panel, ctrl, focus)
                button.Bind(wx.EVT_BUTTON, function)
            elif button_function == "add_input_phase_boxes":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i]: \
                    add_input_phase_boxes(event, panel, ctrl)
                self.textctrl_list[i].Bind(wx.EVT_TEXT, function)
            elif button_function == "add_input_coil_boxes_dyn":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i], index=1: \
                    add_input_coil_boxes(event, panel, ctrl, index)
                self.textctrl_list[i].Bind(wx.EVT_TEXT, function)
            elif button_function == "add_input_coil_boxes_rt":
                function = lambda event, panel=self.panel, ctrl=self.textctrl_list[i], index=2: \
                    add_input_coil_boxes(event, panel, ctrl, index)
                self.textctrl_list[i].Bind(wx.EVT_TEXT, function)

        text_with_button_box.Add(button, 0, wx.ALIGN_LEFT | wx.RIGHT, 10)

        for textctrl in self.textctrl_list:
            text_with_button_box.Add(textctrl, 1, wx.ALIGN_LEFT | wx.LEFT, 10)
            if self.required:
                text_with_button_box.Add(wx.StaticBitmap(self.panel, bitmap=asterisk_icon), 0, wx.RIGHT, 7)

        return text_with_button_box


def on_info_icon_mouse_over(event):
    image = event.GetEventObject()
    image.SetToolTip(wx.ToolTip(image.info_text))


def add_input_coil_boxes(event, tab, ctrl, i=0):
    """On click of ``Number of Custom Coils`` button, add ``n_coils`` ``TextWithButton`` boxes.

    For this function, we are assuming the layout of the Component input is as follows:

        0 - Number of Coils TextWithButton sizer
        1 - Spacer
        2 - next item, and so on

    First, we check and see how many coil boxes the tab currently has, and remove any where
    n current > n update.
    Next, we add n = n update - n current coil boxes to the tab.

    Args:
        event (wx.Event): when the ``Number of Echoes`` button is clicked.
        tab (B0ShimTab): tab class instance for ``B0 Shim``.
        ctrl (wx.TextCtrl): the text box containing the number of phase boxes to add. Must be an
            integer > 0.
        i (int): Index of the coil instance. Used when the tab has multiple coil instances. 1 <= index <= 2
    """

    option_name = "coil"
    try:
        if ctrl.GetValue() == "":
            n_coils = 0
        else:
            n_coils = int(ctrl.GetValue())
        if n_coils < 0:
            raise Exception()
        elif n_coils > 5:
            n_coils = 5
            tab.terminal_component.log_to_terminal("Number of coils limited to 5", level="WARNING")

    except Exception:
        tab.terminal_component.log_to_terminal("Number of coils must be an integer >= 0", level="ERROR")
        n_coils = 0

    # Depending on the index, select the appropriate component
    if i == 1:
        n_coils_displayed = tab.n_coils_dyn
        component_coils = tab.component_coils_dyn
    elif i == 2:
        n_coils_displayed = tab.n_coils_rt
        component_coils = tab.component_coils_rt
    else:
        raise NotImplementedError("Index of the coil instance not implemented for more indexes")

    insert_index = 2
    # If we have to remove coils
    if n_coils < n_coils_displayed:
        for index in range(n_coils_displayed, n_coils, - 1):
            component_coils.sizer.Hide(index + 1)
            component_coils.sizer.Remove(index + 1)
            component_coils.remove_last_input_text_box(option_name)

        # Delete the last spacer if we go back to n_coils == 0
        if n_coils == 0:
            index = 2
            component_coils.sizer.Hide(index)
            component_coils.sizer.Remove(index)

    for index in range(n_coils_displayed, n_coils):
        text_with_button = TextWithButton(
            panel=tab,
            button_label=f"Input Coil {index + 1}",
            button_function=["select_from_overlay", "select_file"],
            default_text="",
            n_text_boxes=2,
            name=f"input_coil_{index + 1}",
            info_text=f"Input path of the coil nifti file followed by json constraint file",
            required=True
        )
        # Add a spacer at the end if its the last one and if there were none previously
        # i.e. if it was previously n_coils == 0
        if index + 1 == n_coils and n_coils_displayed == 0:
            component_coils.insert_input_text_box(
                text_with_button,
                option_name,
                index=insert_index + index,
                last=True)
        else:
            component_coils.insert_input_text_box(
                text_with_button,
                option_name,
                index=insert_index + index
            )

    if i == 1:
        tab.n_coils_dyn = n_coils
    elif i == 2:
        tab.n_coils_rt = n_coils

    tab.SetVirtualSize(tab.sizer_run.GetMinSize())


def add_input_phase_boxes(event, tab, ctrl):
    """On click of ``Number of Echoes`` button, add ``n_echoes`` ``TextWithButton`` boxes.

    For this function, we are assuming the layout of the Component input is as follows:

        0 - Number of Echoes TextWithButton sizer
        1 - Spacer
        2 - next item, and so on

    First, we check and see how many phase boxes the tab currently has, and remove any where
    n current > n update.
    Next, we add n = n update - n current phase boxes to the tab.

    Args:
        event (wx.Event): when the ``Number of Echoes`` button is clicked.
        tab (FieldMapTab): tab class instance for ``Field Map``.
        ctrl (wx.TextCtrl): the text box containing the number of phase boxes to add. Must be an
            integer > 0.
    """
    option_name = "arg"
    try:
        n_echoes = int(ctrl.GetValue())
        if n_echoes < 1:
            raise Exception()
        elif n_echoes > 6:
            n_echoes = 6
            tab.terminal_component.log_to_terminal("Number of echoes limited to 6", level="WARNING")
    except Exception:
        tab.terminal_component.log_to_terminal("Number of Echoes must be an integer > 0", level="ERROR")
        return

    insert_index = 2
    if n_echoes < tab.n_echoes:
        for index in range(tab.n_echoes, n_echoes, - 1):
            tab.component_input.sizer.Hide(index + 1)
            tab.component_input.sizer.Remove(index + 1)
            tab.component_input.remove_last_input_text_box(option_name)

    for index in range(tab.n_echoes, n_echoes):
        text_with_button = TextWithButton(
            panel=tab,
            button_label=f"Input Phase {index + 1}",
            button_function="select_from_overlay",
            default_text="",
            n_text_boxes=1,
            name=f"input_phase_{index + 1}",
            info_text=f"Input path of phase nifti file {index + 1}",
            required=True
        )
        if index + 1 == n_echoes and tab.n_echoes == 0:
            tab.component_input.insert_input_text_box(
                text_with_button,
                option_name,
                index=insert_index + index,
                last=True)
        else:
            tab.component_input.insert_input_text_box(
                text_with_button,
                option_name,
                index=insert_index + index
            )

    tab.n_echoes = n_echoes
    tab.SetVirtualSize(tab.sizer_run.GetMinSize())


class InfoIcon(wx.StaticBitmap):
    def __init__(self, panel, bitmap, info_text):
        self.info_text = info_text
        super(wx.StaticBitmap, self).__init__(panel, bitmap=bitmap)


def create_info_icon(panel, info_text=""):
    image = InfoIcon(panel, bitmap=info_icon, info_text=info_text)
    image.Bind(wx.EVT_MOTION, on_info_icon_mouse_over)
    return image
