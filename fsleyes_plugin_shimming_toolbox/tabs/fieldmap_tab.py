#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__
from fsleyes_plugin_shimming_toolbox.tabs.tab import Tab
from fsleyes_plugin_shimming_toolbox.components.dropdown_component import DropdownComponent
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.components.run_component import RunComponent

from shimmingtoolbox.cli.prepare_fieldmap import prepare_fieldmap_cli


class FieldMapTab(Tab):
    def __init__(self, parent, title="Fieldmap"):
        description = "Create a B0 fieldmap.\n\n" \
                      "Enter the Number of Echoes then press the `Number of Echoes` button.\n\n" \
                      "Select the unwrapper from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.n_echoes = 0
        sizer = self.create_fieldmap_sizer()
        self.sizer_run.Add(sizer, 0, wx.EXPAND)

        self.parent_sizer = self.create_sizer()
        self.SetSizer(self.parent_sizer)

    def create_fieldmap_sizer(self):

        input_text_box_metadata_input = [
            {
                "button_label": "Number of Echoes",
                "button_function": "add_input_phase_boxes",
                "name": "no_arg",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
                "required": True
            }
        ]
        self.component_input = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_input
        )

        dropdown_metadata_unwrapper = [
            {
                "label": "prelude",
                "option_value": "prelude"
            }
        ]
        self.dropdown_unwrapper = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_metadata_unwrapper,
            label="Unwrapper",
            option_name = 'unwrapper',
            cli=prepare_fieldmap_cli
        )

        mask_metadata = [
            {
                "button_label": "Input Mask",
                "button_function": "select_from_overlay",
                "name": "mask",
            }
        ]
        self.component_mask = InputComponent(
            panel=self,
            input_text_box_metadata=mask_metadata,
            cli=prepare_fieldmap_cli
        )

        threshold_metadata = [
            {
                "button_label": "Threshold",
                "name": "threshold",
            },
            {
                "button_label": "Output Calculated Mask",
                "name": "savemask",
                "load_in_overlay": True
            }
        ]
        self.component_threshold = InputComponent(
            panel=self,
            input_text_box_metadata=threshold_metadata,
            cli=prepare_fieldmap_cli
        )

        dropdown_mask_threshold = [
            {
                "label": "auto threshold",
                "option_value": ""
            },
            {
                "label": "mask",
                "option_value": ""
            },
            {
                "label": "threshold",
                "option_value": ""
            },
        ]

        self.dropdown_roi = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_mask_threshold,
            label="Unwrapping region",
            info_text="Masking methods either with a file input or a threshold",
            option_name='no_arg',
            list_components=[self.create_empty_component(), self.component_mask, self.component_threshold],
            cli=prepare_fieldmap_cli
        )

        path_output = os.path.join(__CURR_DIR__, "output_fieldmap")
        input_text_box_metadata_output = [
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "fieldmap.nii.gz"),
                "name": "output",
                "required": True
            }
        ]
        self.component_output = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_output,
            cli=prepare_fieldmap_cli
        )

        input_text_box_metadata_input2 = [
            {
                "button_label": "Input Magnitude",
                "button_function": "select_from_overlay",
                "name": "mag",
                "required": True
            }
        ]
        self.component_input2 = InputComponent(
            panel=self,
            input_text_box_metadata=input_text_box_metadata_input2,
            cli=prepare_fieldmap_cli
        )

        self.run_component = RunComponent(
            panel=self,
            list_components=[self.component_input, self.component_input2, self.dropdown_roi,
                             self.dropdown_unwrapper, self.component_output],
            st_function="st_prepare_fieldmap"
        )

        return self.run_component.sizer
