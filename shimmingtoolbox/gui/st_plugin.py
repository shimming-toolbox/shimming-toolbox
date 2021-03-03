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

import shimmingtoolbox
from shimmingtoolbox import __dir_shimmingtoolbox__
from shimmingtoolbox import gui_utils
from shimmingtoolbox.utils import run_subprocess

import numpy as np
import webbrowser
import nibabel as nib
import os
from pathlib import Path
import tempfile
import logging
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

        # # Toggle off the X and Y canvas
        # oopts = ortho.sceneOpts
        # oopts.showXCanvas = False
        # oopts.showYCanvas = False
        #
        # # Toggle off the cursor
        # oopts.showCursor = False
        #
        # # Toggle off the radiological orientation
        # self.displayCtx.radioOrientation = False
        #
        # # Invert the Y display
        # self.frame.viewPanels[0].frame.viewPanels[0].getZCanvas().opts.invertY = True

        # Create a temporary directory that will hold the NIfTI files
        self.st_temp_dir = tempfile.TemporaryDirectory()

        self.verify_version()

    def load_png_image_from_path(
        self, image_path, is_mask=False, add_to_overlayList=True, colormap="greyscale"
    ):
        """
        This function converts a 2D image into a NIfTI image and loads it as an overlay.
        The parameter add_to_overlayList allows to display the overlay into FSLeyes.
        :param image_path: The location of the image, including the name and the .extension
        :type image_path: string
        :param is_mask: (optional) Whether or not this is a segmentation mask. It will be treated as a normalads_utils
        image by default.
        :type is_mask: bool
        :param add_to_overlayList: (optional) Whether or not to add the image to the overlay list. If so, the image will
        be displayed in the application. This parameter is True by default.
        :type add_to_overlayList: bool
        :param colormap: (optional) the colormap of image that will be displayed. This parameter is set to greyscale by
        default.
        :type colormap: string
        :return: the FSLeyes overlay corresponding to the loaded image.
        :rtype: overlay
        """

        # Open the 2D image
        img_png2D = gui_utils.imread(image_path)

        if is_mask is True:
            img_png2D = img_png2D // params.intensity['binary']  # Segmentation masks should be binary

        # Flip the image on the Y axis so that the morphometrics file shows the right coordinates
        img_png2D = np.flipud(img_png2D)

        # Convert image data into a NIfTI image
        # Note: PIL and NiBabel use different axis conventions, so some array manipulation has to be done.
        img_NIfTI = nib.Nifti1Image(
            np.rot90(img_png2D, k=1, axes=(1, 0)), np.eye(4)
        )

        # Save the NIfTI image in a temporary directory
        img_name = os.path.basename(image_path)
        out_file = self.st_temp_dir.name + "/" + img_name[:-3] + "nii.gz"
        nib.save(img_NIfTI, out_file)

        # Load the NIfTI image as an overlay
        img_overlay = ovLoad.loadOverlays(paths=[out_file], inmem=True, blocking=True)[
            0
        ]

        # Display the overlay
        if add_to_overlayList is True:
            self.overlayList.append(img_overlay)
            opts = self.displayCtx.getOpts(img_overlay)
            opts.cmap = colormap

        return img_overlay

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

        st_path = Path(os.path.abspath(shimmingtoolbox.__file__)).parents[0]
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
            sizer_info | sizer_input | sizer_terminal
        """
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sizer_info)
        sizer.AddSpacer(30)
        sizer.Add(self.sizer_input, wx.EXPAND)
        sizer.AddSpacer(30)
        sizer.Add(self.sizer_terminal, wx.EXPAND)
        return sizer


class InfoComponent:
    def __init__(self, panel, description):
        self.panel = panel
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


class InputComponent:
    def __init__(self, panel, input_text_box_metadata, st_function):
        self.st_function = st_function
        self.sizer = self.create_sizer()
        self.panel = panel
        self.input_text_boxes = {}
        self.input_text_box_metadata = input_text_box_metadata
        self.add_input_text_boxes()
        self.add_button_run()

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer

    def add_input_text_boxes(self, spacer_size=20):
        """Add a list of input text boxes (TextWithButton) to the sizer_input.

        Args:
            TODO: metadata argument does not exist, info is still relevant (uses self.input_text_box_metadata)
            metadata (list)(dict): A list of dictionaries, where the dictionaries have two keys:
                ``button_label`` and ``button_function``.
                .. code::

                    {
                        "button_label": The label to go on the button.
                        "button_function": the class function (self.myfunc) which will get
                            called when the button is pressed. If no action is desired, create
                            a function that is just ``pass``.
                        "default_text": (optional) The default text to be displayed.
                    }
            spacer_size (int): The size of the space to be placed between each input text box.

        """
        for twb_dict in self.input_text_box_metadata:
            text_with_button = TextWithButton(
                panel=self.panel,
                button_label=twb_dict["button_label"],
                button_function=twb_dict.pop("button_function", self.button_do_something),
                default_text=twb_dict.pop("default_text", "")
            )
            self.add_input_text_box(text_with_button, twb_dict.pop("name", "default"))

    def add_input_text_box(self, text_with_button, name, spacer_size=20):
        box = text_with_button.create()
        self.sizer.Add(box, 0, wx.EXPAND)
        self.sizer.AddSpacer(spacer_size)
        self.input_text_boxes[name] = text_with_button

    def add_button_run(self):
        button_run = wx.Button(self.panel, -1, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.button_run_on_click)
        self.sizer.Add(button_run, 0, wx.CENTRE)

    def button_run_on_click(self, event):
        try:
            command, msg = self.get_run_args(self.st_function)
            self.panel.terminal_component.log_to_terminal(msg, level="INFO")
            run_subprocess(command)
        except Exception as err:
            self.panel.terminal_component.log_to_terminal(err, level="ERROR")

    def get_run_args(self, st_function):
        msg = f"Running "
        command = st_function
        for name, input_text_box in self.input_text_boxes.items():
            arg = input_text_box.textctrl.GetValue()
            if arg == "" or arg is None:
                raise RunArgumentErrorST(f"Argument {name} is missing a value, please enter a valid input")
            else:
                command += f" -{name} {arg}"
        msg += command
        return command, msg

    def button_do_something(self, event):
        """TODO"""
        pass


class TerminalComponent:
    def __init__(self, panel):
        self.panel = panel
        self.terminal = None
        self.sizer = self.create_sizer()

    @property
    def terminal(self):
        return self._terminal

    @terminal.setter
    def terminal(self, terminal):
        if terminal is None:
            terminal = wx.TextCtrl(self.panel, wx.ID_ANY, size=(500, 300),
                                   style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
            # Using black background does not change the slider's colour
            # terminal = wx.TextCtrl(self.panel, wx.ID_ANY, size=(500, 300),
            #                        style=wx.TE_MULTILINE | wx.TE_READONLY)
            # terminal.SetDefaultStyle(wx.TextAttr(wx.WHITE, wx.BLACK))
            # terminal.SetBackgroundColour(wx.BLACK)

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

        description = "Shimming Tab description: TODO"
        super().__init__(parent, title, description)

        self.sizer_input = self.create_sizer_input()

        self.create_choice_box()

        self.terminal_component = TerminalComponent(self)
        self.sizer_terminal = self.terminal_component.sizer

        sizer_zshim = self.create_sizer_zshim()
        self.sizer_input.Add(sizer_zshim, 0, wx.EXPAND)

        self.pos_zshim = self.sizer_input.GetItemCount() - 1

        # Create second choice sizer
        sizer_default_text = self.create_sizer_other_algo()
        self.sizer_input.Add(sizer_default_text, 0, wx.EXPAND)
        self.pos_nothing = self.sizer_input.GetItemCount() - 1

        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

        # Run on choice to select the default choice from the choice box widget
        self.on_choice(None)

    def on_choice(self, event):
        # Get the selection from the choice box widget
        selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection == "RT_ZShim":
            sizer_item_zshim = self.sizer_input.GetItem(self.pos_zshim)
            sizer_item_zshim.Show(True)
        elif selection == "Nothing":
            sizer_item_nothing = self.sizer_input.GetItem(self.pos_nothing)
            sizer_item_nothing.Show(True)
        else:
            pass

        # Update the window
        self.Layout()

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        sizer = self.sizer_input.GetItem(self.pos_zshim)
        sizer.Show(False)
        sizer = self.sizer_input.GetItem(self.pos_nothing)
        sizer.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self, choices=["RT_ZShim", "Nothing"])
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_input.Add(self.choice_box)
        self.sizer_input.AddSpacer(10)

    def create_sizer_other_algo(self):
        sizer_shim_default = wx.BoxSizer(wx.VERTICAL)
        description_text = wx.StaticText(self, id=-1, label="Not implemented")
        sizer_shim_default.Add(description_text)
        return sizer_shim_default

    def create_sizer_input(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer

    def create_sizer_zshim(self, metadata=None):
        input_text_box_metadata = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap"
            },
            {
                "button_label": "Input Anat",
                "name": "anat"
            },
            {
                "button_label": "Input Static Mask",
                "name": "mask"
            },
            {
                "button_label": "Input RIRO Mask",
                "name": "mask"
            },
            {
                "button_label": "Input Respiratory Trace",
                "name": "resp"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": __dir_shimmingtoolbox__,
                "name": "output"
            }
        ]
        sizer_zshim = InputComponent(self, input_text_box_metadata, "st_realtime_zshim").sizer
        return sizer_zshim


class FieldMapTab(Tab):
    def __init__(self, parent, title="Field Map"):
        description = "Field Map Tab description: TODO"
        super().__init__(parent, title, description)
        input_text_box_metadata = [
            {
                "button_label": "Number of Echoes"
            },
            {
                "button_label": "Input Echo 1"
            },
            {
                "button_label": "Input Echo 2"
            },
            {
                "button_label": "Input Magnitude",
                "name": "mag"
            },
            {
                "button_label": "Unwrapper",
                "name": "unwrapper"
            },
            {
                "button_label": "Threshold",
                "name": "threshold"
            },
            {
                "button_label": "Input Mask",
                "name": "mask"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": __dir_shimmingtoolbox__,
                "name": "output"
            }
        ]
        self.terminal_component = TerminalComponent(self)
        self.sizer_input = InputComponent(self, input_text_box_metadata, "st_prepare_fieldmap").sizer
        self.sizer_terminal = self.terminal_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)
        # TODO: add CLI for echoes (dropdown?)


class MaskTab(Tab):
    def __init__(self, parent, title="Mask"):
        description = "Mask Tab description: TODO"
        super().__init__(parent, title, description)
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "name": "input"
            },
            {
                "button_label": "Threshold",
                "default_text": "30",
                "name": "thr"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": __dir_shimmingtoolbox__,
                "name": "output"
            }
        ]
        self.terminal_component = TerminalComponent(self)
        self.sizer_input = InputComponent(self, input_text_box_metadata, "st_mask").sizer
        self.sizer_terminal = self.terminal_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)


class DicomToNiftiTab(Tab):
    def __init__(self, parent, title="Dicom to Nifti"):
        description = "Dicom to Nifti Tab description: TODO"
        super().__init__(parent, title, description)
        input_text_box_metadata = [
            {
                "button_label": "Input Folder",
                "button_function": "select_folder",
                "name": "input"
            },
            {
                "button_label": "Subject Name",
                "name": "subject"
            },
            {
                "button_label": "Config Path",
                "button_function": "select_file",
                "default_text": os.path.join(__dir_shimmingtoolbox__,
                                             "config",
                                             "dcm2bids.json"),
                "name": "subject"
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": __dir_shimmingtoolbox__,
                "name": "output"
            }
        ]
        self.terminal_component = TerminalComponent(self)
        self.sizer_input = InputComponent(self, input_text_box_metadata, "st_dicom_to_nifti").sizer
        self.sizer_terminal = self.terminal_component.sizer
        sizer = self.create_sizer()
        self.SetSizer(sizer)


class RunArgumentErrorST(Exception):
    """Exception for missing input arguments for CLI call."""
    pass


class TextWithButton:
    def __init__(self, panel, button_label, button_function, default_text=""):
        self.panel = panel
        self.button_label = button_label
        self.button_function = button_function
        self.default_text = default_text
        self.textctrl = None

    def create(self):
        textctrl = wx.TextCtrl(parent=self.panel, value=self.default_text)
        self.textctrl = textctrl
        text_with_button_box = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self.panel, -1, label=self.button_label)
        if self.button_function == "select_folder":
            self.button_function = lambda event, ctrl=textctrl: select_folder(event, ctrl)
        elif self.button_function == "select_file":
            self.button_function = lambda event, ctrl=textctrl: select_file(event, ctrl)
        button.Bind(wx.EVT_BUTTON, self.button_function)
        text_with_button_box.Add(button, 0, wx.ALIGN_LEFT| wx.RIGHT, 10)
        text_with_button_box.Add(textctrl, 1, wx.ALIGN_LEFT|wx.LEFT, 10)
        return text_with_button_box


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
