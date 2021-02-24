"""
This is an FSLeyes plugin script that integrates shimmingtoolbox tools into FSLeyes.

"""

import wx
import wx.lib.agw.hyperlink as hl

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.actions.loadoverlay as ovLoad

import shimmingtoolbox
from shimmingtoolbox import __dir_shimmingtoolbox__
from shimmingtoolbox import gui_utils

import numpy as np
import webbrowser
import nibabel as nib
from PIL import Image, ImageDraw, ImageOps
import scipy.misc
import os
import json
from pathlib import Path
import math
from scipy import ndimage as ndi
from skimage import measure, morphology, feature
import tempfile
import pandas as pd


VERSION = "0.2.14"


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

        # Toggle off the X and Y canvas
        oopts = ortho.sceneOpts
        oopts.showXCanvas = False
        oopts.showYCanvas = False

        # Toggle off the cursor
        oopts.showCursor = False

        # Toggle off the radiological orientation
        self.displayCtx.radioOrientation = False

        # Invert the Y display
        self.frame.viewPanels[0].frame.viewPanels[0].getZCanvas().opts.invertY = True

        # Create a temporary directory that will hold the NIfTI files
        self.st_temp_dir = tempfile.TemporaryDirectory()

        # self.verify_version()

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
        """
        This function is used to show a popup message on the FSLeyes interface.
        :param message: The message to be displayed.
        :type message: String
        :param caption: (Optional) The caption of the message box.
        :type caption: String
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
        """
        This function checks if the plugin version is the same as the one in the AxonDeepSeg directory
        """
        st_path = Path(os.path.abspath(AxonDeepSeg.__file__)).parents[0]
        plugin_path_parts = st_path.parts[:-1]
        plugin_path = str(Path(*plugin_path_parts))
        plugin_file = plugin_path + "/st_plugin.py"

        # Check if the plugin file exists
        plugin_file_exists = os.path.isfile(plugin_file)

        if plugin_file_exists is False:
            return

        # Check the version of the plugin
        with open(plugin_file) as plugin_file_reader:
            plugin_file_lines = plugin_file_reader.readlines()

        plugin_file_lines = [x.strip() for x in plugin_file_lines]
        version_line = 'VERSION = "' + VERSION + '"'
        plugin_is_up_to_date = True
        version_found = False

        for lines in plugin_file_lines:
            if (lines.startswith("VERSION = ")):
                version_found = True
                if not (lines == version_line):
                    plugin_is_up_to_date = False

        if (version_found is False) or (plugin_is_up_to_date is False):
            message = (
                "A more recent version of the AxonDeepSeg plugin was found in your AxonDeepSeg installation folder. "
                "You will need to replace the current FSLeyes plugin which the new one. "
                "To proceed, go to: file -> load plugin -> ads_plugin.py. Then, restart FSLeyes."
            )
            self.show_message(message, "Warning")
        return

    @staticmethod
    def supportedViews():
        """I am not sure what this method does."""
        from fsleyes.views.orthopanel import OrthoPanel

        return [OrthoPanel]

    @staticmethod
    def defaultLayout():
        """This method makes the control panel appear on the left of the FSLeyes window."""
        return {"location": wx.LEFT}


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
        self.description = description
        self.sizer_info = self.create_sizer_info()

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
            parent=self,
            id=-1,
            bitmap=png,
            pos=wx.DefaultPosition
        )
        return logo_image

    def documentation_url(self, event):
        """Redirect ``documentation_button`` to the ``shimming-toolbox`` page."""
        url = "https://shimming-toolbox.org/en/latest/"
        webbrowser.open(url)

    def create_sizer_info(self):
        """Create the left sizer containing generic Shimming Toolbox information."""
        sizer_info = wx.BoxSizer(wx.VERTICAL)

        st_logo = self.get_logo()
        sizer_info.Add(st_logo, flag=wx.SHAPED, proportion=1)

        button_documentation = wx.Button(self, label="Documentation",
                                         size=wx.Size(100,20))
        button_documentation.Bind(wx.EVT_BUTTON, self.documentation_url)
        sizer_info.Add(button_documentation, flag=wx.SHAPED, proportion=1)

        description_text = wx.StaticText(self, id=-1, label=self.description)
        sizer_info.Add(description_text)
        return sizer_info

    def create_sizer_tab(self):
        """Create the right sizer containing tab-specific functionality."""
        sizer_tab = wx.BoxSizer(wx.VERTICAL)
        sizer_tab.SetMinSize(400, 300)
        sizer_tab.AddSpacer(10)
        return sizer_tab

    def create_sizer(self):
        """Create the parent sizer for the tab.

        Tab is divided into 2 main sizers:
            sizer_info | sizer_tab
        """
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sizer_info)
        sizer.AddSpacer(30)
        sizer.Add(self.sizer_tab, wx.EXPAND)
        return sizer

    def add_input_text_boxes(self, metadata, spacer_size=20):
        """Add a list of input text boxes (TextWithButton) to the sizer_tab.

        Args:
            metadata (list)(dict): A list of dictionaries, where the dictionaries have two keys:
                ``button_label`` and ``button_function``.
                .. code::

                    {
                        "button_label": The label to go on the button.
                        "button function": the class function (self.myfunc) which will get
                            called when the button is pressed. If no action is desired, create
                            a function that is just ``pass``.
                    }
            spacer_size (int): The size of the space to be placed between each input text box.

        """
        for twb_dict in metadata:
            text_with_button = TextWithButton(panel=self,
                                              button_label=twb_dict["button_label"],
                                              button_function=twb_dict["button_function"])
            box = text_with_button.create()
            self.sizer_tab.Add(box, 0, wx.EXPAND)
            self.sizer_tab.AddSpacer(spacer_size)


class ShimTab(Tab):
    def __init__(self, parent, title="Shim"):
        description = "Shimming Tab description: TODO"
        super().__init__(parent, title, description)
        sizer_tab = self.create_sizer_tab()
        self.sizer_tab = sizer_tab
        self.add_input_text_boxes()
        sizer = self.create_sizer()
        self.SetSizer(sizer)

    def add_input_text_boxes(self, metadata=None):
        metadata = [
            {
                "button_label": "Input Fieldmap",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Anat",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Static Mask",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input RIRO Mask",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Respiratory Trace",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Output Folder",
                "button_function": self.button_do_something
            }
        ]
        super().add_input_text_boxes(metadata)

    def button_do_something(self, event):
        """TODO"""
        pass


class FieldMapTab(Tab):
    def __init__(self, parent, title="Field Map"):
        description = "Field Map Tab description: TODO"
        super().__init__(parent, title, description)
        sizer_tab = self.create_sizer_tab()
        self.sizer_tab = sizer_tab
        self.add_input_text_boxes()
        sizer = self.create_sizer()
        self.SetSizer(sizer)

    def add_input_text_boxes(self, metadata=None):
        metadata = [
            {
                "button_label": "Number of Echoes",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Echo 1",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Echo 2",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Magnitude",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Unwrapper",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Threshold",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Input Mask",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Output Folder",
                "button_function": self.button_do_something
            }
        ]
        super().add_input_text_boxes(metadata)

    def button_do_something(self, event):
        """TODO"""
        pass


class MaskTab(Tab):
    def __init__(self, parent, title="Mask"):
        description = "Mask Tab description: TODO"
        super().__init__(parent, title, description)
        sizer_tab = self.create_sizer_tab()
        self.sizer_tab = sizer_tab
        self.add_input_text_boxes()
        sizer = self.create_sizer()
        self.SetSizer(sizer)

    def add_input_text_boxes(self, metadata=None):
        metadata = [
            {
                "button_label": "Input",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Threshold",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Output Folder",
                "button_function": self.button_do_something
            }
        ]
        super().add_input_text_boxes(metadata)

    def button_do_something(self, event):
        """TODO"""
        pass

class DicomToNiftiTab(Tab):
    def __init__(self, parent, title="Dicom to Nifti"):
        description = "Dicom to Nifti Tab description: TODO"
        super().__init__(parent, title, description)
        sizer_tab = self.create_sizer_tab()
        self.sizer_tab = sizer_tab
        self.add_input_text_boxes()
        sizer = self.create_sizer()
        self.SetSizer(sizer)

    def add_input_text_boxes(self, metadata=None):
        metadata = [
            {
                "button_label": "Input Folder",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Subject Name",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Config Path",
                "button_function": self.button_do_something
            },
            {
                "button_label": "Output Folder",
                "button_function": self.button_do_something
            }
        ]
        super().add_input_text_boxes(metadata)

    def button_do_something(self, event):
        """TODO"""
        pass

class TextWithButton:
    def __init__(self, panel, button_label, button_function):
        self.panel = panel
        self.button_label = button_label
        self.button_function = button_function

    def create(self):
        textctrl = wx.TextCtrl(self.panel)
        text_with_button_box = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(self.panel, -1, label=self.button_label)
        button.Bind(wx.EVT_BUTTON, self.button_function)
        text_with_button_box.Add(button, 0, wx.ALIGN_LEFT| wx.RIGHT, 10)
        text_with_button_box.Add(textctrl, 1, wx.ALIGN_LEFT|wx.LEFT, 10)
        return text_with_button_box
