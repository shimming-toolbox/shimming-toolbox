# -*- coding: utf-8 -*-
"""Shimming Toolbox FSLeyes Plugin

This is an FSLeyes plugin script that integrates ``shimmingtoolbox`` tools into FSLeyes:

- dicom_to_nifti_cli
- mask_cli
- prepare_fieldmap_cli
- realtime_zshim_cli

---------------------------------------------------------------------------------------
Copyright (c) 2021 Polytechnique Montreal <www.neuro.polymtl.ca>
Authors: Alexandre D'Astous, Ainsleigh Hill, Charlotte, Gaspard Cereza, Julien Cohen-Adad
"""

import wx

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as ovLoad


import numpy as np
import webbrowser
import nibabel as nib
import os
import abc
import tempfile
import logging
import imageio
import subprocess

__dir_shimmingtoolbox__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger = logging.getLogger(__name__)

VERSION = "0.1.1"


class STControlPanel(ctrlpanel.ControlPanel):
    """Class for Shimming Toolbox Control Panel"""

    def __init__(self, ortho, *args, **kwargs):
        """Initialize the control panel.

        Generates the widgets and adds them to the panel. Also sets the initial position of the
        panel to the left.

        Args:
            ortho: This is used to access the ortho ops in order to turn off the X and Y canvas as
                well as the cursor
        """
        ctrlpanel.ControlPanel.__init__(self, ortho, *args, **kwargs)

        my_panel = TabPanel(self)
        sizer_tabs = wx.BoxSizer(wx.VERTICAL)
        sizer_tabs.SetMinSize(400, 300)
        sizer_tabs.Add(my_panel, 0, wx.EXPAND)

        # Set the sizer of the control panel
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sizer_tabs, wx.EXPAND)
        self.SetSizer(sizer)

        # Initialize the variables that are used to track the active image
        self.png_image_name = []
        self.image_dir_path = []
        self.most_recent_watershed_mask_name = None

        # Create a temporary directory that will hold the NIfTI files
        self.st_temp_dir = tempfile.TemporaryDirectory()

        self.verify_version()

    def show_message(self, message, caption="Error"):
        """Show a popup message on the FSLeyes interface.

        Args:
            message (str): message to be displayed
            caption (str): (optional) caption of the message box.
        """
        with wx.MessageDialog(
            self,
            message,
            caption=caption,
            style=wx.OK | wx.CENTRE,
            pos=wx.DefaultPosition,
        ) as msg:
            msg.ShowModal()

    def verify_version(self):
        """Check if the plugin version is the same as the one in the shimming-toolbox directory."""

        st_path = os.path.realpath(__file__)
        plugin_file = os.path.join(st_path, "gui", "st_plugin.py")

        plugin_file_exists = os.path.isfile(plugin_file)

        if not plugin_file_exists:
            return

        # Check the version of the plugin
        with open(plugin_file) as plugin_file_reader:
            plugin_file_lines = plugin_file_reader.readlines()

        plugin_file_lines = [x.strip() for x in plugin_file_lines]
        version_line = f'VERSION = "{VERSION}"'
        plugin_is_up_to_date = True
        version_found = False

        for lines in plugin_file_lines:
            if lines.startswith("VERSION = "):
                version_found = True
                if not lines == version_line:
                    plugin_is_up_to_date = False

        if version_found is False or plugin_is_up_to_date is False:
            message = """
                A more recent version of the ShimmingToolbox plugin was found in your
                ShimmingToolbox installation folder. You will need to replace the current
                FSLeyes plugin with the new one.
                To proceed, go to: File -> Load plugin -> st_plugin.py. Then, restart FSLeyes.
            """
            self.show_message(message, "Warning")
        return

    @staticmethod
    def supportedViews():
        """I am not sure what this method does."""
        from fsleyes.views.orthopanel import OrthoPanel

        return [OrthoPanel]

    @staticmethod
    def defaultLayout():
        """This method makes the control panel appear on the bottom of the FSLeyes window."""
        return {
            "location": wx.BOTTOM,
            "title": "Shimming Toolbox"
        }


class TabPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent=parent)

        nb = wx.Notebook(self)
        tab1 = ShimTab(nb)
        tab2 = FieldMapTab(nb)
        tab3 = MaskTab(nb)
        tab4 = DicomToNiftiTab(nb)

        # Add the windows to tabs and name them.
        nb.AddPage(tab1, tab1.title)
        nb.AddPage(tab2, tab2.title)
        nb.AddPage(tab3, tab3.title)
        nb.AddPage(tab4, tab4.title)

        sizer = wx.BoxSizer()
        sizer.Add(nb, 1, wx.EXPAND)
        self.SetSizer(sizer)


class Tab(wx.Panel):
    def __init__(self, parent, title, description):
        wx.Panel.__init__(self, parent)
        self.title = title
        self.sizer_info = InfoComponent(self, description).sizer

    def create_sizer(self):
        """Create the parent sizer for the tab.

        Tab is divided into 3 main sizers:
            sizer_info | sizer_run | sizer_terminal
        """
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sizer_info)
        sizer.AddSpacer(30)
        sizer.Add(self.sizer_run, wx.EXPAND)
        sizer.AddSpacer(30)
        sizer.Add(self.sizer_terminal, wx.EXPAND)
        return sizer


class Component:
    def __init__(self, panel, list_components=[]):
        self.panel = panel
        self.list_components = list_components

    @abc.abstractmethod
    def create_sizer(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'create_sizer') and
                callable(subclass.create_sizer) or
                NotImplemented)


class InfoComponent(Component):
    def __init__(self, panel, description):
        super().__init__(panel)
        self.description = description
        self.sizer = self.create_sizer()

    def create_sizer(self):
        """Create the left sizer containing generic Shimming Toolbox information."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        st_logo = self.get_logo()
        sizer.Add(st_logo, flag=wx.SHAPED, proportion=1)

        button_documentation = wx.Button(self.panel, label="Documentation",
                                         size=wx.Size(100, 20))
        button_documentation.Bind(wx.EVT_BUTTON, self.documentation_url)
        sizer.Add(button_documentation, flag=wx.SHAPED, proportion=1)

        description_text = wx.StaticText(self.panel, id=-1, label=self.description)
        width = st_logo.Size[0]
        description_text.Wrap(width)
        sizer.Add(description_text)
        return sizer

    def get_logo(self, scale=0.2):
        """Loads ShimmingToolbox logo saved as a png image and returns it as a wx bitmap image.

        Retunrs:
            wx.StaticBitmap: The ShimmingToolbox logo
        """
        fname_st_logo = os.path.join(__dir_shimmingtoolbox__, 'docs', 'source', '_static',
                                     'shimming_toolbox_logo.png')

        png = wx.Image(fname_st_logo, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        png.SetSize((png.GetWidth()*scale, png.GetHeight()*scale))
        logo_image = wx.StaticBitmap(
            parent=self.panel,
            id=-1,
            bitmap=png,
            pos=wx.DefaultPosition
        )
        return logo_image

    def documentation_url(self, event):
        """Redirect ``documentation_button`` to the ``shimming-toolbox`` page."""
        url = "https://shimming-toolbox.org/en/latest/"
        webbrowser.open(url)


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
        self.sizer.Add(box, 0, wx.EXPAND)
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
        """TODO"""
        pass


class DropdownComponent(Component):
    def __init__(self, panel, dropdown_metadata, name, list_components=[], info_text=""):
        """ Create a dropdown list

        Args:
            panel: A panel is a window on which controls are placed.
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
            info_text (str): Help message when hovering the "i"
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
        for index in range(len(self.dropdown_choices)):
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
        self.input_text_boxes = self.list_components[index].input_text_boxes
        self.input_text_boxes[self.dropdown_metadata[index]["option_name"]] = \
            [self.dropdown_metadata[index]["option_value"]]

        # Update the window
        self.panel.Layout()

    def find_index(self, label):
        for index in range(len(self.dropdown_metadata)):
            if self.dropdown_metadata[index]["label"] == label:
                return index

    def create_sizer(self):
        """Create the a sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer


class RunComponent(Component):
    def __init__(self, panel, st_function, list_components=[], special_output=[]):
        super().__init__(panel, list_components)
        self.st_function = st_function
        self.sizer = self.create_sizer()
        self.add_button_run()
        self.outputs = special_output

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        for component in self.list_components:
            sizer.Add(component.sizer, 0, wx.EXPAND)
        return sizer

    def add_button_run(self):
        button_run = wx.Button(self.panel, -1, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.button_run_on_click)
        self.sizer.Add(button_run, 0, wx.CENTRE)
        self.sizer.AddSpacer(10)

    def button_run_on_click(self, event):
        try:
            command, msg = self.get_run_args(self.st_function)
            self.panel.terminal_component.log_to_terminal(msg, level="INFO")
            run_subprocess(command)
            msg = f"Run {self.st_function} completed successfully"
            self.panel.terminal_component.log_to_terminal(msg, level="INFO")
            self.send_output_to_overlay()
        except Exception as err:
            self.panel.terminal_component.log_to_terminal(str(err), level="ERROR")

    def send_output_to_overlay(self):
        for i_file in self.outputs:
            if os.path.isfile(i_file):
                try:
                    # Display the overlay
                    window = self.panel.GetGrandParent().GetParent()
                    if i_file[-4:] == ".png":
                        load_png_image_from_path(window, i_file, colormap="greyscale")
                    elif i_file[-7:] == ".nii.gz" or i_file[-4:] == ".nii":
                        # Load the NIfTI image as an overlay
                        img_overlay = ovLoad.loadOverlays(paths=[i_file], inmem=True, blocking=True)[0]
                        window.overlayList.append(img_overlay)
                except Exception as err:
                    self.panel.terminal_component.log_to_terminal(str(err), level="ERROR")

    def get_run_args(self, st_function):
        msg = "Running "
        command = st_function
        # Init arguments and options
        command_list_arguments = []
        command_dict_options = {}
        for component in self.list_components:
            for name, input_text_box_list in component.input_text_boxes.items():
                if name == "no_arg":
                    continue
                for input_text_box in input_text_box_list:
                    # Allows to chose from a dropdown
                    if type(input_text_box) == str:
                        if name in command_dict_options.keys():
                            command_dict_options[name].append(input_text_box)
                        else:
                            command_dict_options[name] = [input_text_box]
                    # Normal case where input_text_box is a TextwithButton
                    else:
                        for textctrl in input_text_box.textctrl_list:
                            arg = textctrl.GetValue()
                            if arg == "" or arg is None:
                                if input_text_box.required is True:
                                    raise RunArgumentErrorST(
                                        f"Argument {name} is missing a value, please enter a valid input"
                                    )
                            else:
                                # Case where the option name is set to arg, this handles it as if it were an argument
                                if name == "arg":
                                    command_list_arguments.append(arg)
                                # Normal options
                                else:
                                    if name == "output":
                                        self.outputs.append(arg)
                                    if name in command_dict_options.keys():
                                        command_dict_options[name].append(arg)
                                    else:
                                        command_dict_options[name] = [arg]

        # Arguments don't need "-"
        for arg in command_list_arguments:
            command += f" {arg}"

        # Handles options
        for name, args in command_dict_options.items():
            command += f" -{name}"
            for arg in args:
                command += f" {arg}"
        msg += command
        return command, msg


class TerminalComponent(Component):
    def __init__(self, panel, list_components=[]):
        super().__init__(panel, list_components)
        self.terminal = None
        self.sizer = self.create_sizer()

    @property
    def terminal(self):
        return self._terminal

    @terminal.setter
    def terminal(self, terminal):
        if terminal is None:
            terminal = wx.TextCtrl(self.panel, wx.ID_ANY, size=(500, 300),
                                   style=wx.TE_MULTILINE | wx.TE_READONLY)
            terminal.SetDefaultStyle(wx.TextAttr(wx.WHITE, wx.BLACK))
            terminal.SetBackgroundColour(wx.BLACK)

        self._terminal = terminal

    def create_sizer(self):
        """Create the right sizer containing the terminal interface."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddSpacer(10)
        sizer.Add(self.terminal)
        return sizer

    def log_to_terminal(self, msg, level=None):
        if level is None:
            self.terminal.AppendText(f"{msg}\n")
        else:
            self.terminal.AppendText(f"{level}: {msg}\n")


class ShimTab(Tab):
    def __init__(self, parent, title="Shim"):

        description = "Perform B0 shimming.\n\n" \
                      "Select the shimming algorithm from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.positions = {}
        self.dropdown_metadata = [
            {
                "name": "RT_ZShim",
                "sizer_function": self.create_sizer_zshim
            },
            {
                "name": "Nothing",
                "sizer_function": self.create_sizer_other_algo
            }
        ]
        self.dropdown_choices = [item["name"] for item in self.dropdown_metadata]

        self.create_choice_box()

        self.terminal_component = TerminalComponent(self)
        self.sizer_terminal = self.terminal_component.sizer

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

    def create_sizer_zshim(self, metadata=None):
        path_output = os.path.join(__dir_shimmingtoolbox__, "output_rt_zshim")
        input_text_box_metadata = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap",
                "button_function": "select_from_overlay",
                "info_text": "B0 fieldmap. This should be a 4D file (4th dimension being time).",
                "required": True
            },
            {
                "button_label": "Input Anat",
                "name": "anat",
                "button_function": "select_from_overlay",
                "info_text": "Filename of the anatomical image to apply the correction.",
                "required": True
            },
            {
                "button_label": "Input Static Mask",
                "name": "mask-static",
                "button_function": "select_from_overlay",
                "info_text": """3D NIfTI file used to define the static spatial region to shim.
                    The coordinate system should be the same as anat's coordinate system."""
            },
            {
                "button_label": "Input RIRO Mask",
                "name": "mask-riro",
                "button_function": "select_from_overlay",
                "info_text": """3D NIfTI file used to define the time varying (i.e. RIRO,
                    Respiration-Induced Resonance Offset) spatial region to shim.
                    The coordinate system should be the same as anat's coordinate system."""
            },
            {
                "button_label": "Input Respiratory Trace",
                "button_function": "select_file",
                "name": "resp",
                "info_text": "Siemens respiratory file containing pressure data.",
                "required": True
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
                "info_text": "Directory to output gradient text file and figures."
            }
        ]

        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_realtime_zshim",
            special_output=[
                os.path.join(path_output, "fig_resampled_riro.nii.gz"),
                os.path.join(path_output, "fig_resampled_static.nii.gz")
            ]
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_other_algo(self):
        sizer_shim_default = wx.BoxSizer(wx.VERTICAL)
        description_text = wx.StaticText(self, id=-1, label="Not implemented")
        sizer_shim_default.Add(description_text)
        return sizer_shim_default

    def create_sizer_run(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer


class FieldMapTab(Tab):
    def __init__(self, parent, title="Field Map"):
        description = "Create a B0 fieldmap.\n\n" \
                      "Enter the number of echoes then press the `Number of Echoes` button.\n\n" \
                      "Select the unwrapper from the dropdown list."
        super().__init__(parent, title, description)
        self.n_echoes = 0
        input_text_box_metadata_input = [
            {
                "button_label": "Number of Echoes",
                "button_function": "add_input_echo_boxes",
                "name": "no_arg",
                "info_text": "Number of echo NIfTI files to be used. Must be an integer > 0.",
                "required": True
            }
        ]
        dropdown_metadata = [
            {
                "label": "prelude",
                "option_name": "unwrapper",
                "option_value": "prelude"
            },
            {
                "label": "Nothing",
                "option_name": "unwrapper",
                "option_value": "QGU"
            }
        ]
        input_text_box_metadata_prelude = [
            {
                "button_label": "Input Magnitude",
                "button_function": "select_from_overlay",
                "name": "mag",
                "info_text": "Input path of mag NIfTI file.",
                "required": True
            },
            {
                "button_label": "Threshold",
                "name": "threshold",
                "info_text": "Float threshold for masking. Used for: PRELUDE."
            },
            {
                "button_label": "Input Mask",
                "button_function": "select_from_overlay",
                "name": "mask",
                "info_text": "Input path for a mask. Used for PRELUDE"
            }
        ]
        input_text_box_metadata_other = [
            {
                "button_label": "Other",
                "name": "other",
                "info_text": "TODO"
            }
        ]
        input_text_box_metadata_output = [
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(
                    __dir_shimmingtoolbox__,
                    "output_fieldmap",
                    "fieldmap.nii.gz"),
                "name": "output",
                "info_text": "Output filename for the fieldmap, supported types : '.nii', '.nii.gz'",
                "required": True
            }
        ]

        self.terminal_component = TerminalComponent(panel=self)
        self.component_input = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_input
        )
        self.component_prelude = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_prelude
        )
        self.component_other = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_other
        )
        self.dropdown = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_metadata,
            list_components=[self.component_prelude, self.component_other],
            name="Unwrapper",
            info_text="Algorithm for unwrapping"
        )
        self.component_output = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_output
        )
        self.run_component = RunComponent(
            panel=self,
            list_components=[self.component_input, self.dropdown, self.component_output],
            st_function="st_prepare_fieldmap"
        )
        self.sizer_run = self.run_component.sizer
        self.sizer_terminal = self.terminal_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)


class MaskTab(Tab):
    def __init__(self, parent, title="Mask"):
        description = "Create a mask based.\n\n" \
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

        self.terminal_component = TerminalComponent(self)
        self.sizer_terminal = self.terminal_component.sizer

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
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": """Input path of the nifti file to mask. Supported extensions are
                    .nii or .nii.gz.""",
                "required": True
            },
            {
                "button_label": "Threshold",
                "default_text": "30",
                "name": "thr",
                "info_text": """Integer value to threshold the data: voxels will be set to zero if
                    their value <= this threshold. Default = 30."""
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(
                    __dir_shimmingtoolbox__,
                    "output_mask_threshold",
                    "mask.nii.gz"
                ),
                "name": "output",
                "info_text": """Name of output mask. Supported extensions are .nii or .nii.gz."""
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
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": """Input path of the NIfTI file to mask. The NIfTI file must be 2D or
                    3D. Supported extensions are .nii or .nii.gz.""",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 2,
                "info_text": "Length of the side of the box along 1st & 2nd dimension (in pixels).",
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 2,
                "info_text": """Center of the box along first and second dimension (in pixels).
                    If no center is provided (None), the middle is used."""
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": os.path.join(
                    __dir_shimmingtoolbox__,
                    "output_mask_rect",
                    "mask.nii.gz"
                ),
                "name": "output",
                "info_text": """Name of output mask. Supported extensions are .nii or .nii.gz."""
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
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "info_text": """Input path of the NIfTI file to mask. The NIfTI file must be 3D.
                    Supported extensions are .nii or .nii.gz.""",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 3,
                "info_text": "Length of side of box along 1st, 2nd, & 3rd dimension (in pixels).",
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 3,
                "info_text": """Center of the box along 1st, 2nd, & 3rd dimension (in pixels).
                    If no center is provided (None), the middle is used."""
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": os.path.join(
                    __dir_shimmingtoolbox__,
                    "output_mask_box",
                    "mask.nii.gz"
                ),
                "name": "output",
                "info_text": """Name of output mask. Supported extensions are .nii or .nii.gz."""
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
        description = "Process dicoms into NIfTI following the BIDS data structure"
        super().__init__(parent, title, description)
        input_text_box_metadata = [
            {
                "button_label": "Input Folder",
                "button_function": "select_folder",
                "name": "input",
                "info_text": "Input path of dicom folder",
                "required": True
            },
            {
                "button_label": "Subject Name",
                "name": "subject",
                "info_text": "Name of the patient",
                "required": True
            },
            {
                "button_label": "Config Path",
                "button_function": "select_file",
                "default_text": os.path.join(__dir_shimmingtoolbox__,
                                             "config",
                                             "dcm2bids.json"),
                "name": "config",
                "info_text": "Full file path and name of the BIDS config file"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": os.path.join(__dir_shimmingtoolbox__, "output_dicom_to_nifti"),
                "name": "output",
                "info_text": "Output path for NIfTI files."
            }
        ]
        self.terminal_component = TerminalComponent(self)
        component = InputComponent(self, input_text_box_metadata)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_dicom_to_nifti"
        )
        self.sizer_run = run_component.sizer
        self.sizer_terminal = self.terminal_component.sizer
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

        panel: TODO
        button_label (str): label to be put on the button.
        button_function: function which gets called when the button is clicked on.
        default_text (str): (optional) default text to be displayed in the input text box.
        textctrl_list (list wx.TextCtrl): list of input text boxes, can be more than one in a row.
        n_text_boxes (int): number of input text boxes to create.
        name (str): TODO
        info_text (str): text to be displayed when hovering over the info icon; should describe
            what the button/input is for.
        required (bool): if this input is required or not. If True, a red asterisk will be
            placed next to the input text box to indicate this.
    """
    def __init__(self, panel, button_label, button_function, name="default", default_text="",
                 n_text_boxes=1, info_text="", required=False):
        self.panel = panel
        self.button_label = button_label
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
            if i_text_box == 0:
                if self.button_function == "select_folder":
                    self.button_function = lambda event, ctrl=textctrl: select_folder(event, ctrl)
                elif self.button_function == "select_file":
                    self.button_function = lambda event, ctrl=textctrl: select_file(event, ctrl)
                elif self.button_function == "select_from_overlay":
                    self.button_function = lambda event, panel=self.panel, ctrl=textctrl: \
                        select_from_overlay(event, panel, ctrl)
                elif self.button_function == "add_input_echo_boxes":
                    self.button_function = lambda event, panel=self.panel, ctrl=textctrl: \
                        add_input_echo_boxes(event, panel, ctrl)
                    textctrl.Bind(wx.EVT_TEXT, self.button_function)
                button.Bind(wx.EVT_BUTTON, self.button_function)
                text_with_button_box.Add(button, 0, wx.ALIGN_LEFT | wx.RIGHT, 10)

            text_with_button_box.Add(textctrl, 1, wx.ALIGN_LEFT | wx.LEFT, 10)
            if self.required:
                text_with_button_box.Add(
                    create_asterisk_icon(self.panel), 0, wx.ALIGN_RIGHT | wx.RIGHT, 7
                )

        return text_with_button_box


def create_asterisk_icon(panel):
    bmp = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION)
    info_icon = os.path.join(__dir_shimmingtoolbox__, 'shimmingtoolbox', 'gui', 'asterisk.png')
    img = wx.Image(info_icon, wx.BITMAP_TYPE_ANY)
    bmp = img.ConvertToBitmap()
    image = wx.StaticBitmap(panel, bitmap=bmp)
    return image


def create_info_icon(panel, info_text=""):
    bmp = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION)
    info_icon = os.path.join(__dir_shimmingtoolbox__, 'shimmingtoolbox', 'gui', 'info-icon.png')
    img = wx.Image(info_icon, wx.BITMAP_TYPE_ANY)
    bmp = img.ConvertToBitmap()
    image = InfoIcon(panel, bitmap=bmp, info_text=info_text)
    image.Bind(wx.EVT_MOTION, on_info_icon_mouse_over)
    return image


def on_info_icon_mouse_over(event):
    image = event.GetEventObject()
    tooltip = wx.ToolTip(image.info_text)
    tooltip.SetDelay(10)
    image.SetToolTip(tooltip)


class InfoIcon(wx.StaticBitmap):
    def __init__(self, panel, bitmap, info_text):
        self.info_text = info_text
        super(wx.StaticBitmap, self).__init__(panel, bitmap=bitmap)


def select_folder(event, ctrl):
    """Select a file folder from system path."""
    dlg = wx.DirDialog(None, "Choose Directory", __dir_shimmingtoolbox__,
                       wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() == wx.ID_OK:
        folder = dlg.GetPath()
        ctrl.SetValue(folder)
        logger.info(f"Folder set to: {folder}")


def select_file(event, ctrl):
    """Select a file from system path."""
    dlg = wx.FileDialog(parent=None,
                        message="Select File",
                        defaultDir=__dir_shimmingtoolbox__,
                        style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)

    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        ctrl.SetValue(path)
        logger.info(f"File set to: {path}")


def select_from_overlay(event, tab, ctrl):
    """Fetch path to file highlighted in the Overlay list.

    Args:
        event (wx.Event): event passed to a callback or member function.
        tab (Tab): Must be a subclass of the Tab class
        ctrl (wx.TextCtrl): the text item.
    """

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


def add_input_echo_boxes(event, tab, ctrl):
    """On click of ``Number of Echoes`` button, add ``n_echoes`` ``TextWithButton`` boxes.

    For this function, we are assuming the layout of the Component input is as follows:

        0 - Number of Echoes TextWithButton sizer
        1 - Spacer
        2 - next item, and so on

    First, we check and see how many echo boxes the tab currently has, and remove any where
    n current > n update.
    Next, we add n = n update - n current echo boxes to the tab.

    Args:
        event (wx.Event): when the ``Number of Echoes`` button is clicked.
        tab (FieldMapTab): tab class instance for ``Field Map``.
        ctrl (wx.TextCtrl): the text box containing the number of echo boxes to add. Must be an
            integer > 0.
    """
    option_name = "arg"
    try:
        n_echoes = int(ctrl.GetValue())
        if n_echoes < 1:
            raise Exception()
    except Exception:
        tab.terminal_component.log_to_terminal(
            "Number of Echoes must be an integer > 0",
            level="ERROR"
        )
        return

    insert_index = 2
    if n_echoes < tab.n_echoes:
        for index in range(tab.n_echoes, n_echoes, -1):
            tab.component_input.sizer.Hide(index + 1)
            tab.component_input.sizer.Remove(index + 1)
            tab.component_input.remove_last_input_text_box(option_name)

    for index in range(tab.n_echoes, n_echoes):
        text_with_button = TextWithButton(
            panel=tab,
            button_label=f"Input Echo {index + 1}",
            button_function="select_from_overlay",
            default_text="",
            n_text_boxes=1,
            name=f"input_echo_{index + 1}",
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
    tab.Layout()


class RunArgumentErrorST(Exception):
    """Exception for missing input arguments for CLI call."""
    pass


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


# TODO: find a better way to include this as it is defined in utils as well
def run_subprocess(cmd):
    """Wrapper for ``subprocess.run()`` that enables to input ``cmd`` as a full string (easier for debugging).

    Args:
        cmd (string): full command to be run on the command line
    """
    logging.debug(f'{cmd}')
    try:
        subprocess.run(
            cmd.split(' '),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as err:
        msg = "Return code: ", err.returncode, "\nOutput: ", err.stderr
        raise Exception(msg)


def load_png_image_from_path(fsl_panel, image_path, is_mask=False, add_to_overlayList=True,
                             colormap="greyscale"):
    """Converts a 2D image into a NIfTI image and loads it as an overlay.
    The parameter add_to_overlayList allows to display the overlay into FSLeyes.

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
    img_overlay = ovLoad.loadOverlays(paths=[fname_out], inmem=True, blocking=True)[
        0
    ]

    # Display the overlay
    if add_to_overlayList is True:
        fsl_panel.overlayList.append(img_overlay)
        opts = fsl_panel.displayCtx.getOpts(img_overlay)
        opts.cmap = colormap

    return img_overlay
