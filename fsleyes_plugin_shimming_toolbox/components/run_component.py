#!/usr/bin/python3
# -*- coding: utf-8 -*

import fsleyes.actions.loadoverlay as loadoverlay
import glob
import imageio
import nibabel as nib
import numpy as np
import os
import wx

from fsleyes_plugin_shimming_toolbox import __DIR_ST_PLUGIN_IMG__
from fsleyes_plugin_shimming_toolbox.components.component import Component, RunArgumentErrorST
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.events import EVT_RESULT, EVT_LOG
from fsleyes_plugin_shimming_toolbox.worker_thread import WorkerThread


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
        self.load_in_overlay = []
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
        play_icon = wx.Bitmap(os.path.join(__DIR_ST_PLUGIN_IMG__, 'play.png'), wx.BITMAP_TYPE_PNG)
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
        # Return code is 0 if everything ran smoothly
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

            # Append files that are a direct output from "load_in_overlay"
            for fname in self.load_in_overlay:
                self.output_paths.append(fname)

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
            self.panel.terminal_component.log_to_terminal(str(data), level="ERROR")

        self.worker = None
        self.load_in_overlay = []
        event.Skip()

    def button_run_on_click(self, event):
        """Function called when the ``Run`` button is clicked.

        Calls the relevant ``Shimming Toolbox`` CLI command (``st_function``) in a thread

        """
        self.run()

    def run(self):
        if not self.worker:
            try:
                command, msg = self.get_run_args(self.st_function)
            except RunArgumentErrorST as err:
                self.panel.terminal_component.log_to_terminal(err, level="ERROR")
                return

            self.panel.terminal_component.log_to_terminal(msg, level="INFO")
            self.worker = WorkerThread(self.panel, command, name=self.st_function)

    def send_output_to_overlay(self):
        for output_path in self.output_paths:
            if os.path.isfile(output_path):
                try:
                    # Display the overlay
                    window = self.panel.GetGrandParent()
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
        # Split is necessary if we have grouped commands (st_mask threshold)
        command = st_function.split(' ')
    
        for component in self.list_components:  
            
            cmd, output, load_in_overlay = component.get_command()
            command.extend(cmd)
            
            if st_function.split(' ')[-1] == "realtime-dynamic" and cmd[0] == "--coil":
                cmd_riro = ['--coil-riro' if i == '--coil' else i for i in cmd]
                command.extend(cmd_riro)
                
            self.load_in_overlay.extend(load_in_overlay)
            
            if output:
                self.output = output
                
        msg += ' '.join(command) + '\n'

        return command, msg

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
