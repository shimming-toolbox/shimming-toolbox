#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__, __ST_DIR__
from fsleyes_plugin_shimming_toolbox.tabs.tab import Tab
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.components.run_component import RunComponent

from shimmingtoolbox.cli.dicom_to_nifti import dicom_to_nifti_cli


class DicomToNiftiTab(Tab):
    def __init__(self, parent, title="Dicom to Nifti"):
        description = "Convert DICOM files into NIfTI following the BIDS data structure"
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.run_component = None
        sizer = self.create_dicom_to_nifti_sizer()
        self.sizer_run.Add(sizer, 0, wx.EXPAND)

        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

    def create_dicom_to_nifti_sizer(self):
        path_output = os.path.join(__CURR_DIR__, "output_dicom_to_nifti")
        input_text_box_metadata = [
            {
                "button_label": "Input Folder",
                "button_function": "select_folder",
                "name": "input",
                "required": True
            },
            {
                "button_label": "Subject Name",
                "name": "subject",
                "required": True
            },
            {
                "button_label": "Config Path",
                "button_function": "select_file",
                "default_text": os.path.join(__ST_DIR__, "dcm2bids.json"),
                "name": "config",
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, cli=dicom_to_nifti_cli)
        self.run_component = RunComponent(panel=self, list_components=[component], st_function="st_dicom_to_nifti")
        return self.run_component.sizer
