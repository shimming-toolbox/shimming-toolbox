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

import abc
import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as loadoverlay
import fsleyes.views.canvaspanel as canvaspanel
import glob
import imageio
import logging
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import webbrowser
import wx

from fsleyes_plugin_shimming_toolbox.events import result_event_type, EVT_RESULT, ResultEvent
from fsleyes_plugin_shimming_toolbox.events import log_event_type, EVT_LOG, LogEvent
from fsleyes_plugin_shimming_toolbox.worker_thread import WorkerThread
from shimmingtoolbox.cli.b0shim import dynamic_cli, realtime_cli
from shimmingtoolbox.cli.b1shim import b1shim_cli
from shimmingtoolbox.cli.dicom_to_nifti import dicom_to_nifti_cli
from shimmingtoolbox.cli.mask import box, rect, threshold
from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli

logger = logging.getLogger(__name__)

HOME_DIR = str(Path.home())
CURR_DIR = os.getcwd()
ST_DIR = f"{HOME_DIR}/shimming-toolbox"
DIR = os.path.dirname(__file__)

VERSION = "0.1.1"

# Load icon resources
asterisk_icon = wx.Image(os.path.join(DIR, 'img', 'asterisk.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap()
info_icon = wx.Image(os.path.join(DIR, 'img', 'info-icon.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap()
play_icon = wx.Bitmap(os.path.join(DIR, 'img', 'play.png'), wx.BITMAP_TYPE_PNG)
rtd_logo = wx.Bitmap(os.path.join(DIR, 'img', 'RTD.png'), wx.BITMAP_TYPE_PNG)
# Load ShimmingToolbox logo saved as a png image, rescale it, and return it as a wx.Bitmap image.
st_logo = wx.Image(os.path.join(DIR, 'img', 'shimming_toolbox_logo.png'), wx.BITMAP_TYPE_PNG)
st_logo.Rescale(st_logo.GetWidth() * 0.2, st_logo.GetHeight() * 0.2, wx.IMAGE_QUALITY_HIGH)
st_logo = st_logo.ConvertToBitmap()


# We need to create a ctrlpanel.ControlPanel instance so that it can be recognized as a plugin by FSLeyes
# Class hierarchy: wx.Panel > fslpanel.FSLeyesPanel > ctrlpanel.ControlPanel
class STControlPanel(ctrlpanel.ControlPanel):
    """Class for Shimming Toolbox Control Panel"""

    # The CanvasPanel view is used for most FSLeyes plugins so we decided to stick to it
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


class Tab(wx.ScrolledWindow):
    def __init__(self, parent, title, description):
        super().__init__(parent)
        self.title = title
        self.sizer_info = InfoSection(self, description).sizer
        self.terminal_component = parent.terminal_component
        self.SetScrollbars(1, 4, 1, 1)

    def create_sizer(self):
        """Create the parent sizer for the tab.

        Tab is divided into 2 main sizers:
            sizer_info | sizer_run
        """
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sizer_info, 0)
        sizer.AddSpacer(20)
        sizer.Add(self.sizer_run, 2)
        return sizer

    def create_empty_component(self):
        component = InputComponent(panel=self, input_text_box_metadata=[])
        return component


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


class Component:
    def __init__(self, panel, list_components=[]):
        self.panel = panel
        self.list_components = list_components

    @abc.abstractmethod
    def create_sizer(self):
        raise NotImplementedError

    # make sure that the create_sizer method has been implemented in the subclasses
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'create_sizer') and callable(subclass.create_sizer) or NotImplemented


class InfoSection:
    def __init__(self, panel, description):
        self.panel = panel
        self.description = description
        self.sizer = self.create_sizer()

    def create_sizer(self):
        """Create the left sizer containing generic Shimming Toolbox information."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        logo = wx.StaticBitmap(parent=self.panel, id=-1, bitmap=st_logo, pos=wx.DefaultPosition)
        width = logo.Size[0]
        sizer.Add(logo, flag=wx.SHAPED, proportion=1)
        sizer.AddSpacer(10)

        # Create a "Documentation" button that redirects towards https://shimming-toolbox.org/en/latest/
        button_documentation = wx.Button(self.panel, label="Documentation")
        button_documentation.Bind(wx.EVT_BUTTON, self.open_documentation_url)
        button_documentation.SetBitmap(rtd_logo)
        sizer.Add(button_documentation, flag=wx.EXPAND)
        sizer.AddSpacer(10)

        # Add a short description of what the current tab does
        description_text = wx.StaticText(self.panel, id=-1, label=self.description)
        description_text.Wrap(width)
        sizer.Add(description_text)
        return sizer

    def open_documentation_url(self, event):
        """Redirect ``button_documentation`` to the ``shimming-toolbox`` page."""
        webbrowser.open('https://shimming-toolbox.org/en/latest/')


class InputComponent(Component):
    def __init__(self, panel, input_text_box_metadata):
        super().__init__(panel)
        self.sizer = self.create_sizer()
        self.input_text_boxes = {}
        self.input_text_box_metadata = input_text_box_metadata
        self.add_input_text_boxes()

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer

    def add_input_text_boxes(self, spacer_size=10):
        """Add a list of input text boxes (TextWithButton) to the sizer_input.

        Args:
            self.input_text_box_metadata (list)(dict): A list of dictionaries, where the dictionaries have two keys:
                ``button_label`` and ``button_function``.
                .. code::

                    {
                        "button_label": The label to go on the button.
                        "button_function": the class function (self.myfunc) which will get
                            called when the button is pressed. If no action is desired, create
                            a function that is just ``pass``.
                        "default_text": (optional) The default text to be displayed.
                        "name" : Option name in the CLI, use "arg" as the name for an argument.
                    }

            spacer_size (int): The size of the space to be placed between each input text box.

        """
        for twb_dict in self.input_text_box_metadata:
            text_with_button = TextWithButton(
                panel=self.panel,
                button_label=twb_dict["button_label"],
                button_function=twb_dict.get("button_function", self.button_do_something),
                default_text=twb_dict.get("default_text", ""),
                n_text_boxes=twb_dict.get("n_text_boxes", 1),
                name=twb_dict.get("name", "default"),
                info_text=twb_dict.get("info_text", ""),
                required=twb_dict.get("required", False)
            )
            self.add_input_text_box(text_with_button, twb_dict.get("name", "default"))

    def add_input_text_box(self, text_with_button, name, spacer_size=10):
        box = text_with_button.create()
        self.sizer.Add(box, 1, wx.EXPAND)
        self.sizer.AddSpacer(spacer_size)
        if name in self.input_text_boxes.keys():
            self.input_text_boxes[name].append(text_with_button)
        else:
            self.input_text_boxes[name] = [text_with_button]

    def insert_input_text_box(self, text_with_button, name, index, last=False, spacer_size=10):
        box = text_with_button.create()
        self.sizer.Insert(index=index, sizer=box, flag=wx.EXPAND)
        if last:
            self.sizer.InsertSpacer(index=index + 1, size=spacer_size)
        if name in self.input_text_boxes.keys():
            self.input_text_boxes[name].append(text_with_button)
        else:
            self.input_text_boxes[name] = [text_with_button]

    def remove_last_input_text_box(self, name):
        self.input_text_boxes[name].pop(-1)

    def button_do_something(self, event):
        pass


class DropdownComponent(Component):
    def __init__(self, panel, dropdown_metadata, name, list_components=[], info_text=""):
        """ Create a dropdown list

        Args:
            panel (wx.Panel): A panel is a window on which controls are placed.
            dropdown_metadata (list)(dict): A list of dictionaries where the dictionaries have the
                required keys: ``label``, ``option_name``, ``option_value``.
                .. code::

                    {
                        "label": The label for the dropdown box
                        "option_name": The name of the option in the CLI
                        "option_value": The value linked to the option in the CLI
                    }

            name (str): Label of the button describing the dropdown
            list_components (list): list of InputComponents
            info_text (str): Info message displayed when hovering over the "i" icon.
        """
        super().__init__(panel, list_components)
        self.dropdown_metadata = dropdown_metadata
        self.name = name
        self.info_text = info_text
        self.positions = {}
        self.input_text_boxes = {}
        self.sizer = self.create_sizer()
        self.dropdown_choices = [item["label"] for item in self.dropdown_metadata]
        self.create_choice_box()
        self.create_dropdown_sizers()
        self.on_choice(None)

    def create_dropdown_sizers(self):
        for index in range(len(self.list_components)):
            sizer = self.list_components[index].sizer
            self.sizer.Add(sizer, 0, wx.EXPAND)
            self.positions[self.dropdown_choices[index]] = self.sizer.GetItemCount() - 1

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        for position in self.positions.values():
            sizer = self.sizer.GetItem(position)
            sizer.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self.panel, choices=self.dropdown_choices)
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        button = wx.Button(self.panel, -1, label=self.name)
        self.choice_box_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.choice_box_sizer.Add(create_info_icon(self.panel, self.info_text), 0, wx.ALIGN_LEFT | wx.RIGHT, 7)
        self.choice_box_sizer.Add(button, 0, wx.ALIGN_LEFT | wx.RIGHT, 10)
        self.choice_box_sizer.Add(self.choice_box)
        self.sizer.Add(self.choice_box_sizer)
        self.sizer.AddSpacer(10)

    def on_choice(self, event):
        # Get the selection from the choice box widget
        selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            sizer_item_threshold = self.sizer.GetItem(self.positions[selection])
            sizer_item_threshold.Show(True)
        else:
            pass

        index = self.find_index(selection)
        if selection in self.positions.keys():
            # Add the sizers to the current list of options
            self.input_text_boxes = self.list_components[index].input_text_boxes

        # Add the dropdown to the list of options
        self.input_text_boxes[self.dropdown_metadata[index]["option_name"]] = \
            [self.dropdown_metadata[index]["option_value"]]
        # Update the window
        self.panel.Layout()

    def find_index(self, label):
        for index in range(len(self.dropdown_metadata)):
            if self.dropdown_metadata[index]["label"] == label:
                return index

        # Return index 0 if it is not found
        return 0

    def create_sizer(self):
        """Create the a sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer


class RunComponent(Component):
    """Component which contains input and run button.

    Attributes:
        panel (wx.Panel): Panel, this is usually a Tab instance
        st_function (str): Name of the ``Shimming Toolbox`` CLI function to be called.
        list_components (list of Component): list of subcomponents to be added.
        output_paths (list of str): file or folder paths containing output from ``st_function``.
        output_paths (list of str): relative path of files containing output from ``st_function``. Path is relative to
                                    the output option's folder.
    """

    def __init__(self, panel, st_function, list_components=[], output_paths=[]):
        super().__init__(panel, list_components)
        self.st_function = st_function
        self.sizer = self.create_sizer()
        self.add_button_run()
        self.output = ""
        self.output_paths_original = output_paths
        self.output_paths = output_paths.copy()
        self.worker = None

        self.panel.Bind(EVT_RESULT, self.on_result)
        self.panel.Bind(EVT_LOG, self.log)

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        for component in self.list_components:
            sizer.Add(component.sizer, 0, wx.EXPAND)
        return sizer

    def add_button_run(self):
        """Add the run button which will call the ``Shimming Toolbox`` CLI."""
        button_run = wx.Button(self.panel, -1, label="Run", size=(85, 48))
        button_run.Bind(wx.EVT_BUTTON, self.button_run_on_click)
        button_run.SetBitmap(play_icon, dir=wx.LEFT)
        self.sizer.Add(button_run, 0, wx.CENTRE)
        self.sizer.AddSpacer(10)

    def log(self, event):
        """Log to the terminal the when there is a log event"""

        # Since the log events get broadcated to all the RunComponents of the Tab, we check that the event name
        # corresponds to our function i.e self.st_function
        if event.name == self.st_function:
            msg = event.get_data()
            self.panel.terminal_component.log_to_terminal(msg)
        else:
            event.Skip()

    def on_result(self, event):
        # Since the log events get broadcated to all the RunComponents of the Tab, we check that the event name
        # corresponds to our function i.e self.st_function
        if event.name != self.st_function:
            event.Skip()
            return

        data = event.get_data()
        if data == 0:
            msg = f"Run {self.st_function} completed successfully\n"
            self.panel.terminal_component.log_to_terminal(msg, level="INFO")

            # Get the directory of the output if it is a file or already a directory
            if os.path.isfile(self.output):
                folder = os.path.split(self.output)[0]
            else:
                folder = self.output

            # Add the directory to the relative path of path output
            for i_file in range(len(self.output_paths)):
                self.output_paths[i_file] = os.path.join(folder, self.output_paths[i_file])

            # Append the file if it was a file
            if os.path.isfile(self.output):
                self.output_paths.append(self.output)

            if self.st_function == "st_dicom_to_nifti":
                # If its dicom_to_nifti, output all .nii found in the subject folder to the overlay
                try:
                    path_output, subject = self.fetch_paths_dicom_to_nifti()
                    path_sub = os.path.join(path_output, 'sub-' + subject)
                    list_files = sorted(glob.glob(os.path.join(path_sub, '*', '*.nii*')))
                    for file in list_files:
                        self.output_paths.append(file)

                except Exception:
                    self.panel.terminal_component.log_to_terminal(
                        "Could not fetch subject and/or path to load to overlay"
                    )
            self.send_output_to_overlay()

            self.output_paths.clear()
            self.output_paths = self.output_paths_original.copy()

        elif type(data) == Exception:
            msg = f"Run {self.st_function} errored out\n"
            err = data

            self.panel.terminal_component.log_to_terminal(msg, level="ERROR")
            if len(err.args) == 1:
                # Pretty output
                a_string = ""
                for i_err in range(len(err.args[0])):
                    a_string += str(err.args[0][i_err])
                self.panel.terminal_component.log_to_terminal(a_string, level="ERROR")

            else:
                self.panel.terminal_component.log_to_terminal(str(err), level="ERROR")

        else:
            # The error message should already be displayed
            self.panel.terminal_component.log_to_terminal("")

        self.worker = None
        event.Skip()

    def button_run_on_click(self, event):
        """Function called when the ``Run`` button is clicked.

        Calls the relevant ``Shimming Toolbox`` CLI command (``st_function``) in a thread

        """
        if not self.worker:
            command, msg = self.get_run_args(self.st_function)
            self.panel.terminal_component.log_to_terminal(msg, level="INFO")
            self.worker = WorkerThread(self.panel, command, name=self.st_function)

    def send_output_to_overlay(self):
        for output_path in self.output_paths:
            if os.path.isfile(output_path):
                try:
                    # Display the overlay
                    window = self.panel.GetGrandParent().GetParent()
                    if output_path[-4:] == ".png":
                        load_png_image_from_path(window, output_path, colormap="greyscale")
                    elif output_path[-7:] == ".nii.gz" or output_path[-4:] == ".nii":
                        # Load the NIfTI image as an overlay
                        img_overlay = loadoverlay.loadOverlays(
                            paths=[output_path],
                            inmem=True,
                            blocking=True)[0]
                        window.overlayList.append(img_overlay)
                except Exception as err:
                    self.panel.terminal_component.log_to_terminal(str(err), level="ERROR")

    def get_run_args(self, st_function):
        """The option are a list of tuples where the tuple: (name, [value1, value2])"""
        msg = "Running "
        command = st_function

        self.output = ""
        command_list_arguments = []
        command_list_options = []
        for component in self.list_components:
            for name, input_text_box_list in component.input_text_boxes.items():
                if name == "no_arg":
                    continue

                for input_text_box in input_text_box_list:
                    # Allows to chose from a dropdown
                    if type(input_text_box) == str:
                        command_list_options.append((name, [input_text_box]))

                    # Normal case where input_text_box is a TextwithButton
                    else:
                        is_arg = False
                        option_values = []
                        for textctrl in input_text_box.textctrl_list:
                            arg = textctrl.GetValue()
                            if arg == "" or arg is None:
                                if input_text_box.required is True:
                                    raise self.RunArgumentErrorST(
                                        f"Argument {name} is missing a value, please enter a valid input"
                                    )
                            else:
                                # Case where the option name is set to arg, this handles it as if it were an argument
                                if name == "arg":
                                    command_list_arguments.append(arg)
                                    is_arg = True
                                # Normal options
                                else:
                                    if name == "output":
                                        self.output = arg

                                    option_values.append(arg)

                        # If its an argument don't include it as an option, if the option list is empty don't either
                        if not is_arg and option_values:
                            command_list_options.append((name, option_values))

        # Arguments don't need "-"
        for arg in command_list_arguments:
            command += f" {arg}"

        # Handles options
        for name, args in command_list_options:
            command += f" --{name}"
            for arg in args:
                command += f" {arg}"
        msg += command
        msg += "\n"
        return command, msg

    class RunArgumentErrorST(Exception):
        """Exception for missing input arguments for CLI call."""
        pass

    def fetch_paths_dicom_to_nifti(self):
        if self.st_function == "st_dicom_to_nifti":

            for component in self.list_components:

                if 'subject' in component.input_text_boxes.keys():
                    box_with_button_sub = component.input_text_boxes['subject'][0]
                    subject = box_with_button_sub.textctrl_list[0].GetValue()

                if 'output' in component.input_text_boxes.keys():
                    box_with_button_out = component.input_text_boxes['output'][0]
                    path_output = box_with_button_out.textctrl_list[0].GetValue()

            return path_output, subject


class B0ShimTab(Tab):
    def __init__(self, parent, title="B0 Shim"):

        description = "Perform B0 shimming.\n\n" \
                      "Select the shimming algorithm from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.positions = {}
        self.dropdown_metadata = [
            {
                "name": "Dynamic",
                "sizer_function": self.create_sizer_dynamic_shim
            },
            {
                "name": "Realtime Dynamic",
                "sizer_function": self.create_sizer_realtime_shim
            },
        ]
        self.dropdown_choices = [item["name"] for item in self.dropdown_metadata]

        # Dyn + rt shim
        self.n_coils_rt = 0
        self.n_coils_dyn = 0
        self.component_coils_dyn = None
        self.component_coils_rt = None

        self.create_choice_box()

        self.create_dropdown_sizers()
        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

        # Run on choice to select the default choice from the choice box widget
        self.on_choice(None)

    def create_dropdown_sizers(self):
        for dropdown_dict in self.dropdown_metadata:
            sizer = dropdown_dict["sizer_function"]()
            self.sizer_run.Add(sizer, 0, wx.EXPAND)
            self.positions[dropdown_dict["name"]] = self.sizer_run.GetItemCount() - 1

    def on_choice(self, event):
        # Get the selection from the choice box widget
        selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            run_component_sizer_item = self.sizer_run.GetItem(self.positions[selection])
            run_component_sizer_item.Show(True)

            # When doing Show(True), we show everything in the sizer, we need to call the dropdowns that can contain
            # items to show the appropriate things according to their current choice.
            if selection == 'Dynamic':
                self.dropdown_slice_dyn.on_choice(None)
            elif selection == 'Realtime Dynamic':
                self.dropdown_slice_rt.on_choice(None)
        else:
            pass

        # Update the window
        self.Layout()

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        for position in self.positions.values():
            sizer_item = self.sizer_run.GetItem(position)
            sizer_item.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self, choices=self.dropdown_choices)
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_run.Add(self.choice_box)
        self.sizer_run.AddSpacer(10)

    def create_sizer_dynamic_shim(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "output_dynamic_shim")

        # no_arg is used here since a --coil option must be used for each of the coils (defined add_input_coil_boxes)
        input_text_box_metadata_coil = [
            {
                "button_label": "Number of Custom Coils",
                "button_function": "add_input_coil_boxes_dyn",
                "name": "no_arg",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
            }
        ]
        self.component_coils_dyn = InputComponent(self, input_text_box_metadata_coil)

        input_text_box_metadata_inputs = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap",
                "button_function": "select_from_overlay",
                "info_text": f"{dynamic_cli.params[1].help}",
                "required": True
            },
            {
                "button_label": "Input Anat",
                "name": "anat",
                "button_function": "select_from_overlay",
                "info_text": f"{dynamic_cli.params[2].help}",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
                "info_text": f"{dynamic_cli.params[3].help}"
            },
            {
                "button_label": "Mask Dilation Kernel Size",
                "name": "mask-dilation-kernel-size",
                "info_text": f"{dynamic_cli.params[9].help}",
                "default_text": "3",
            }
        ]

        component_inputs = InputComponent(self, input_text_box_metadata_inputs)

        input_text_box_metadata_scanner = [
            {
                "button_label": "Scanner constraints",
                "button_function": "select_file",
                "name": "scanner-coil-constraints",
                "info_text": f"{dynamic_cli.params[5].help}",
                "default_text": f"{os.path.join(ST_DIR, 'coil_config.json')}",
            },
        ]
        component_scanner = InputComponent(self, input_text_box_metadata_scanner)

        input_text_box_metadata_slice = [
            {
                "button_label": "Slice Factor",
                "name": "slice-factor",
                "info_text": f"{dynamic_cli.params[7].help}",
                "default_text": "1",
            },
        ]
        component_slice_int = InputComponent(self, input_text_box_metadata_slice)
        component_slice_seq = InputComponent(self, input_text_box_metadata_slice)

        output_metadata = [
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{dynamic_cli.params[10].help}"
            }
        ]
        component_output = InputComponent(self, output_metadata)

        dropdown_scanner_order_metadata = [
            {
                "label": "-1",
                "option_name": "scanner-coil-order",
                "option_value": "-1"
            },
            {
                "label": "0",
                "option_name": "scanner-coil-order",
                "option_value": "0"
            },
            {
                "label": "1",
                "option_name": "scanner-coil-order",
                "option_value": "1"
            },
            {
                "label": "2",
                "option_name": "scanner-coil-order",
                "option_value": "2"
            }
        ]

        dropdown_scanner_order = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_order_metadata,
            name="Scanner Order",
            info_text=f"{dynamic_cli.params[4].help}"
        )

        dropdown_ovf_metadata = [
            {
                "label": "delta",
                "option_name": "output-value-format",
                "option_value": "delta"
            },
            {
                "label": "absolute",
                "option_name": "output-value-format",
                "option_value": "absolute"
            }
        ]

        dropdown_ovf = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_ovf_metadata,
            name="Output Value Format",
            info_text=f"{dynamic_cli.params[13].help}"
        )

        dropdown_opt_metadata = [
            {
                "label": "Least Squares",
                "option_name": "optimizer-method",
                "option_value": "least_squares"
            },
            {
                "label": "Pseudo Inverse",
                "option_name": "optimizer-method",
                "option_value": "pseudo_inverse"
            },
        ]

        dropdown_opt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_opt_metadata,
            name="Optimizer",
            info_text=f"{dynamic_cli.params[8].help}"
        )

        dropdown_slice_metadata = [
            {
                "label": "Sequential",
                "option_name": "slices",
                "option_value": "sequential"
            },
            {
                "label": "Interleaved",
                "option_name": "slices",
                "option_value": "interleaved"
            },
            {
                "label": "Volume",
                "option_name": "slices",
                "option_value": "volume"
            },
        ]

        self.dropdown_slice_dyn = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_slice_metadata,
            name="Slice Ordering",
            info_text="Defines the slice ordering.",
            list_components=[component_slice_int, component_slice_seq, self.create_empty_component()]
        )

        dropdown_coil_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_name": "output-file-format-coil",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Slicewise per Coil",
                "option_name": "output-file-format-coil",
                "option_value": "slicewise-coil"
            },
            {
                "label": "Chronological per Channel",
                "option_name": "output-file-format-coil",
                "option_value": "chronological-ch"
            },
            {
                "label": "Chronological per Coil",
                "option_name": "output-file-format-coil",
                "option_value": "chronological-coil"
            },
        ]

        dropdown_coil_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_coil_format_metadata,
            name="Custom Coil Output Format",
            info_text=f"{dynamic_cli.params[11].help}"
        )

        dropdown_scanner_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Slicewise per Coil",
                "option_name": "output-file-format-scanner",
                "option_value": "slicewise-coil"
            },
            {
                "label": "Chronological per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "chronological-ch"
            },
            {
                "label": "Chronological per Coil",
                "option_name": "output-file-format-scanner",
                "option_value": "chronological-coil"
            },
            {
                "label": "Gradient per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "gradient"
            },
        ]

        dropdown_scanner_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_format_metadata,
            name="Scanner Output Format",
            info_text=f"{dynamic_cli.params[12].help}"
        )

        run_component = RunComponent(
            panel=self,
            list_components=[self.component_coils_dyn, component_inputs, dropdown_opt, self.dropdown_slice_dyn,
                             dropdown_scanner_order, component_scanner, dropdown_scanner_format, dropdown_coil_format,
                             dropdown_ovf, component_output],
            st_function="st_b0shim dynamic",
            output_paths=["fieldmap_calculated_shim_masked.nii.gz",
                          "fieldmap_calculated_shim.nii.gz"]
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_realtime_shim(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "output_realtime_shim")

        # no_arg is used here since a --coil option must be used for each of the coils (defined add_input_coil_boxes)
        input_text_box_metadata_coil = [
            {
                "button_label": "Number of Custom Coils",
                "button_function": "add_input_coil_boxes_rt",
                "name": "no_arg",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
            }
        ]
        self.component_coils_rt = InputComponent(self, input_text_box_metadata_coil)

        input_text_box_metadata_inputs = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap",
                "button_function": "select_from_overlay",
                "info_text": f"{realtime_cli.params[1].help}",
                "required": True
            },
            {
                "button_label": "Input Anat",
                "name": "anat",
                "button_function": "select_from_overlay",
                "info_text": f"{realtime_cli.params[2].help}",
                "required": True
            },
            {
                "button_label": "Input Respiratory Trace",
                "name": "resp",
                "button_function": "select_file",
                "info_text": f"{realtime_cli.params[3].help}",
                "required": True
            },
            {
                "button_label": "Input Mask Static",
                "name": "mask-static",
                "button_function": "select_from_overlay",
                "info_text": f"{realtime_cli.params[4].help}"
            },
            {
                "button_label": "Input Mask Realtime",
                "name": "mask-riro",
                "button_function": "select_from_overlay",
                "info_text": f"{realtime_cli.params[5].help}"
            },
            {
                "button_label": "Mask Dilation Kernel Size",
                "name": "mask-dilation-kernel-size",
                "info_text": f"{realtime_cli.params[11].help}",
                "default_text": "3",
            }
        ]

        component_inputs = InputComponent(self, input_text_box_metadata_inputs)

        input_text_box_metadata_scanner = [
            {
                "button_label": "Scanner constraints",
                "button_function": "select_file",
                "name": "scanner-coil-constraints",
                "info_text": f"{realtime_cli.params[7].help}",
                "default_text": f"{os.path.join(ST_DIR, 'coil_config.json')}",
            },
        ]
        component_scanner = InputComponent(self, input_text_box_metadata_scanner)

        input_text_box_metadata_slice = [
            {
                "button_label": "Slice Factor",
                "name": "slice-factor",
                "info_text": f"{realtime_cli.params[9].help}",
                "default_text": "1",
            },
        ]
        component_slice_int = InputComponent(self, input_text_box_metadata_slice)
        component_slice_seq = InputComponent(self, input_text_box_metadata_slice)

        output_metadata = [
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{realtime_cli.params[12].help}"
            }
        ]
        component_output = InputComponent(self, output_metadata)

        dropdown_scanner_order_metadata = [
            {
                "label": "-1",
                "option_name": "scanner-coil-order",
                "option_value": "-1"
            },
            {
                "label": "0",
                "option_name": "scanner-coil-order",
                "option_value": "0"
            },
            {
                "label": "1",
                "option_name": "scanner-coil-order",
                "option_value": "1"
            },
            {
                "label": "2",
                "option_name": "scanner-coil-order",
                "option_value": "2"
            }
        ]

        dropdown_scanner_order = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_order_metadata,
            name="Scanner Order",
            info_text=f"{realtime_cli.params[6].help}"
        )

        dropdown_ovf_metadata = [
            {
                "label": "delta",
                "option_name": "output-value-format",
                "option_value": "delta"
            },
            {
                "label": "absolute",
                "option_name": "output-value-format",
                "option_value": "absolute"
            }
        ]

        dropdown_ovf = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_ovf_metadata,
            name="Output Value Format",
            info_text=f"{realtime_cli.params[15].help}"
        )

        dropdown_opt_metadata = [
            {
                "label": "Least Squares",
                "option_name": "optimizer-method",
                "option_value": "least_squares"
            },
            {
                "label": "Pseudo Inverse",
                "option_name": "optimizer-method",
                "option_value": "pseudo_inverse"
            },
        ]

        dropdown_opt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_opt_metadata,
            name="Optimizer",
            info_text=f"{realtime_cli.params[10].help}"
        )

        dropdown_slice_metadata = [
            {
                "label": "Sequential",
                "option_name": "slices",
                "option_value": "sequential"
            },
            {
                "label": "Interleaved",
                "option_name": "slices",
                "option_value": "interleaved"
            },
            {
                "label": "Volume",
                "option_name": "slices",
                "option_value": "volume"
            },
        ]

        self.dropdown_slice_rt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_slice_metadata,
            name="Slice Ordering",
            info_text=f"{realtime_cli.params[8].help}",
            list_components=[component_slice_int, component_slice_seq, self.create_empty_component()]
        )

        dropdown_coil_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_name": "output-file-format-coil",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Chronological per Channel",
                "option_name": "output-file-format-coil",
                "option_value": "chronological-ch"
            }
        ]

        dropdown_coil_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_coil_format_metadata,
            name="Custom Coil Output Format",
            info_text=f"{realtime_cli.params[13].help}"
        )

        dropdown_scanner_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Chronological per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "chronological-ch"
            },
            {
                "label": "Gradient per Channel",
                "option_name": "output-file-format-scanner",
                "option_value": "gradient"
            },
        ]

        dropdown_scanner_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_format_metadata,
            name="Scanner Output Format",
            info_text=f"{realtime_cli.params[14].help}"
        )

        run_component = RunComponent(
            panel=self,
            list_components=[self.component_coils_rt, component_inputs, dropdown_opt, self.dropdown_slice_rt,
                             dropdown_scanner_order, component_scanner, dropdown_scanner_format,
                             dropdown_coil_format, dropdown_ovf, component_output],
            st_function="st_b0shim realtime-dynamic",
            # TODO: output paths
            output_paths=[]
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_run(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer


class B1ShimTab(Tab):
    def __init__(self, parent, title=r"B1+ Shim"):

        description = "Perform B1+ shimming.\n\n" \
                      "Select the shimming algorithm from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.positions = {}
        self.dropdown_metadata = [
            {
                "name": "CV reduction",
                "sizer_function": self.create_sizer_cv
            },
            {
                "name": "Target",
                "sizer_function": self.create_sizer_target
            },
            {
                "name": "SAR efficiency",
                "sizer_function": self.create_sizer_sar_eff
            },
            {
                "name": "Phase-only",
                "sizer_function": self.create_sizer_phase_only
            }
        ]
        self.dropdown_choices = [item["name"] for item in self.dropdown_metadata]

        self.create_choice_box()

        self.create_dropdown_sizers()
        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

        # Run on choice to select the default choice from the choice box widget
        self.on_choice(None)

    def create_dropdown_sizers(self):
        for dropdown_dict in self.dropdown_metadata:
            sizer = dropdown_dict["sizer_function"]()
            self.sizer_run.Add(sizer, 0, wx.EXPAND)
            self.positions[dropdown_dict["name"]] = self.sizer_run.GetItemCount() - 1

    def on_choice(self, event):
        # Get the selection from the choice box widget
        selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            sizer_item = self.sizer_run.GetItem(self.positions[selection])
            sizer_item.Show(True)
        else:
            pass

        # Update the window
        self.Layout()

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        for position in self.positions.values():
            sizer_item = self.sizer_run.GetItem(position)
            sizer_item.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self, choices=self.dropdown_choices)
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_run.Add(self.choice_box)
        self.sizer_run.AddSpacer(10)

    def create_sizer_cv(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[0].help}",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[1].help}"
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
                "info_text": f"{b1shim_cli.params[4].help}"
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
                "info_text": f"{b1shim_cli.params[5].help}"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{b1shim_cli.params[6].help}"
            }
        ]

        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 1",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_target(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[0].help}",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[1].help}"
            },
            {
                "button_label": "Target value (nT/V)",
                "name": "target",
                "default_text": "20",
                "info_text": f"{b1shim_cli.params[3].help}",
                "required": True
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
                "info_text": f"{b1shim_cli.params[4].help}"
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
                "info_text": f"{b1shim_cli.params[5].help}"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{b1shim_cli.params[6].help}"
            }
        ]

        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 2",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_sar_eff(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[0].help}",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[1].help}"
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
                "info_text": f"{b1shim_cli.params[4].help}",
                "required": True
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
                "info_text": f"{b1shim_cli.params[5].help}"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{b1shim_cli.params[6].help}"
            }
        ]

        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 3",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_phase_only(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ maps",
                "name": "b1",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[0].help}",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
                "info_text": f"{b1shim_cli.params[1].help}"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{b1shim_cli.params[6].help}"
            }
        ]
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 4",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_run(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer


class FieldMapTab(Tab):
    def __init__(self, parent, title="Fieldmap"):
        description = "Create a B0 fieldmap.\n\n" \
                      "Enter the Number of Echoes then press the `Number of Echoes` button.\n\n" \
                      "Select the unwrapper from the dropdown list."
        super().__init__(parent, title, description)
        self.n_echoes = 0
        input_text_box_metadata_input = [
            {
                "button_label": "Number of Echoes",
                "button_function": "add_input_phase_boxes",
                "name": "no_arg",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
                "required": True
            }
        ]
        dropdown_metadata = [
            {
                "label": "prelude",
                "option_name": "unwrapper",
                "option_value": "prelude"
            }
        ]

        dropdown_mask_threshold = [
            {
                "label": "mask",
                "option_name": "no_arg",
                "option_value": ""
            },
            {
                "label": "threshold",
                "option_name": "no_arg",
                "option_value": ""
            },
        ]

        path_output = os.path.join(CURR_DIR, "output_fieldmap")

        input_text_box_metadata_output = [
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "fieldmap.nii.gz"),
                "name": "output",
                "info_text": f"{prepare_fieldmap_cli.params[3].help}",
                "required": True
            }
        ]

        mask_metadata = [
            {
                "button_label": "Input Mask",
                "button_function": "select_from_overlay",
                "name": "mask",
                "info_text": f"{prepare_fieldmap_cli.params[5].help}"
            }
        ]

        self.component_mask = InputComponent(
            panel=self,
            input_text_box_metadata=mask_metadata
        )

        threshold_metadata = [
            {
                "button_label": "Threshold",
                "name": "threshold",
                "info_text": f"{prepare_fieldmap_cli.params[6].help}"
            }
        ]

        self.component_threshold = InputComponent(
            panel=self,
            input_text_box_metadata=threshold_metadata
        )

        self.dropdown_mask_threshold = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_mask_threshold,
            name="Mask/Threshold",
            info_text="Masking methods either with a file input or a threshold",
            list_components=[self.component_mask, self.component_threshold]
        )

        input_text_box_metadata_input2 = [
            {
                "button_label": "Input Magnitude",
                "button_function": "select_from_overlay",
                "name": "mag",
                "info_text": f"{prepare_fieldmap_cli.params[1].help}",
                "required": True
            }
        ]
        self.component_input = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_input
        )
        self.component_input2 = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_input2
        )
        self.dropdown = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_metadata,
            name="Unwrapper",
            info_text=f"{prepare_fieldmap_cli.params[2].help}"
        )
        self.component_output = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_output
        )
        self.run_component = RunComponent(
            panel=self,
            list_components=[self.component_input, self.component_input2, self.dropdown_mask_threshold, self.dropdown,
                             self.component_output],
            st_function="st_prepare_fieldmap"
        )
        self.sizer_run = self.run_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)


class MaskTab(Tab):
    def __init__(self, parent, title="Mask"):
        description = "Create a mask.\n\n" \
                      "Select a shape or an algorithm from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.positions = {}
        self.dropdown_metadata = [
            {
                "name": "Threshold",
                "sizer_function": self.create_sizer_threshold
            },
            {
                "name": "Rectangle",
                "sizer_function": self.create_sizer_rect
            },
            {
                "name": "Box",
                "sizer_function": self.create_sizer_box
            }
        ]
        self.dropdown_choices = [item["name"] for item in self.dropdown_metadata]
        self.create_choice_box()

        self.create_dropdown_sizers()
        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

        # Run on choice to select the default choice from the choice box widget
        self.on_choice(None)

    def create_dropdown_sizers(self):
        for dropdown_dict in self.dropdown_metadata:
            sizer = dropdown_dict["sizer_function"]()
            self.sizer_run.Add(sizer, 0, wx.EXPAND)
            self.positions[dropdown_dict["name"]] = self.sizer_run.GetItemCount() - 1

    def on_choice(self, event):
        # Get the selection from the choice box widget
        selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            sizer_item = self.sizer_run.GetItem(self.positions[selection])
            sizer_item.Show(True)
        else:
            pass

        # Update the window
        self.Layout()
        self.GetParent().Layout()

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        for position in self.positions.values():
            sizer = self.sizer_run.GetItem(position)
            sizer.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self, choices=self.dropdown_choices)
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_run.Add(self.choice_box)
        self.sizer_run.AddSpacer(10)

    def create_sizer_threshold(self, metadata=None):
        path_output = os.path.join(CURR_DIR, "output_mask_threshold")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": f"{threshold.params[0].help}",
                "required": True
            },
            {
                "button_label": "Threshold",
                "default_text": "30",
                "name": "thr",
                "info_text": f"{threshold.params[2].help}"
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
                "info_text": f"{threshold.params[1].help}"
            }
        ]
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask threshold"
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_rect(self):
        path_output = os.path.join(CURR_DIR, "output_mask_rect")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": f"{rect.params[0].help}",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 2,
                "info_text": f"{rect.params[2].help}",
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 2,
                "info_text": f"{rect.params[3].help}"
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
                "info_text": f"{rect.params[1].help}"
            }
        ]
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask rect"
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_box(self):
        path_output = os.path.join(CURR_DIR, "output_mask_box")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": f"{box.params[0].help}",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 3,
                "info_text": f"{box.params[2].help}",
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 3,
                "info_text": f"{box.params[3].help}"
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
                "info_text": f"{box.params[1].help}"
            }
        ]
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask box"
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_run(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer


class DicomToNiftiTab(Tab):
    def __init__(self, parent, title="Dicom to Nifti"):
        description = "Convert DICOM files into NIfTI following the BIDS data structure"
        super().__init__(parent, title, description)
        path_output = os.path.join(CURR_DIR, "output_dicom_to_nifti")
        input_text_box_metadata = [
            {
                "button_label": "Input Folder",
                "button_function": "select_folder",
                "name": "input",
                "info_text": f"{dicom_to_nifti_cli.params[0].help}",
                "required": True
            },
            {
                "button_label": "Subject Name",
                "name": "subject",
                "info_text": f"{dicom_to_nifti_cli.params[1].help}",
                "required": True
            },
            {
                "button_label": "Config Path",
                "button_function": "select_file",
                "default_text": os.path.join(ST_DIR, "dcm2bids.json"),
                "name": "config",
                "info_text": f"{dicom_to_nifti_cli.params[3].help}"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": f"{dicom_to_nifti_cli.params[2].help}"
            }
        ]
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(panel=self, list_components=[component], st_function="st_dicom_to_nifti")
        self.sizer_run = run_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)


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
                 n_text_boxes=1, info_text="", required=False):
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


def create_info_icon(panel, info_text=""):
    image = InfoIcon(panel, bitmap=info_icon, info_text=info_text)
    image.Bind(wx.EVT_MOTION, on_info_icon_mouse_over)
    return image


def on_info_icon_mouse_over(event):
    image = event.GetEventObject()
    image.SetToolTip(wx.ToolTip(image.info_text))


class InfoIcon(wx.StaticBitmap):
    def __init__(self, panel, bitmap, info_text):
        self.info_text = info_text
        super(wx.StaticBitmap, self).__init__(panel, bitmap=bitmap)


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

    dlg = wx.DirDialog(None, "Choose Directory", CURR_DIR, wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

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
                        defaultDir=CURR_DIR,
                        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        ctrl.SetValue(path)
        logger.info(f"File set to: {path}")

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
    window = tab.GetGrandParent().GetParent()
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


def read_image(filename, bitdepth=8):
    """Read image and convert it to desired bitdepth without truncation."""
    if 'tif' in str(filename):
        raw_img = imageio.imread(filename, format='tiff-pil')
        if len(raw_img.shape) > 2:
            raw_img = imageio.imread(filename, format='tiff-pil', as_gray=True)
    else:
        raw_img = imageio.imread(filename)
        if len(raw_img.shape) > 2:
            raw_img = imageio.imread(filename, as_gray=True)

    img = imageio.core.image_as_uint(raw_img, bitdepth=bitdepth)
    return img


def write_image(filename, img, format='png'):
    """Write image."""
    imageio.imwrite(filename, img, format=format)


def load_png_image_from_path(fsl_panel, image_path, is_mask=False, add_to_overlayList=True, colormap="greyscale"):
    """Convert a 2D image into a NIfTI image and load it as an overlay.

    The parameter ``add_to_overlayList`` enables displaying the overlay in FSLeyes.

    Args:
        image_path (str): The location of the image, including the name and the .extension
        is_mask (bool): (optional) Whether or not this is a segmentation mask. It will be
            treated as a normalads_utils
        add_to_overlayList (bool): (optional) Whether or not to add the image to the overlay
            list. If so, the image will be displayed in the application. This parameter is
            True by default.
        colormap (str): (optional) the colormap of image that will be displayed. This parameter
            is set to greyscale by default.

    Returns:
        overlay: the FSLeyes overlay corresponding to the loaded image.
    """

    # Open the 2D image
    img_png2d = read_image(image_path)

    if is_mask is True:
        img_png2d = img_png2d // np.iinfo(np.uint8).max  # Segmentation masks should be binary

    # Flip the image on the Y axis so that the morphometrics file shows the right coordinates
    img_png2d = np.flipud(img_png2d)

    # Convert image data into a NIfTI image
    # Note: PIL and NiBabel use different axis conventions, so some array manipulation has to be done.
    # TODO: save in the FOV of the current overlay
    nii_img = nib.Nifti1Image(
        np.rot90(img_png2d, k=1, axes=(1, 0)), np.eye(4)
    )

    # Save the NIfTI image in a temporary directory
    fname_out = image_path[:-3] + "nii.gz"
    nib.save(nii_img, fname_out)

    # Load the NIfTI image as an overlay
    img_overlay = loadoverlay.loadOverlays(paths=[fname_out], inmem=True, blocking=True)[0]

    # Display the overlay
    if add_to_overlayList is True:
        fsl_panel.overlayList.append(img_overlay)
        opts = fsl_panel.displayCtx.getOpts(img_overlay)
        opts.cmap = colormap

    return img_overlay
