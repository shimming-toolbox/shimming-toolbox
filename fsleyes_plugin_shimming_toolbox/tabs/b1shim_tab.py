#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__
from fsleyes_plugin_shimming_toolbox.tabs.tab import Tab
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.components.run_component import RunComponent

from shimmingtoolbox.cli.b1shim import b1shim_cli


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
        if self.choice_box.GetSelection() < 0:
            selection = self.choice_box.GetString(0)
            self.choice_box.SetSelection(0)
        else:
            selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            sizer_item = self.sizer_run.GetItem(self.positions[selection])
            sizer_item.Show(True)
        else:
            pass

        # Update the window
        self.SetVirtualSize(self.sizer_run.GetMinSize())
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
        path_output = os.path.join(__CURR_DIR__, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]

        component = InputComponent(self, input_text_box_metadata, cli=b1shim_cli)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 1",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_target(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Target value (nT/V)",
                "name": "target",
                "default_text": "20",
                "required": True
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]

        component = InputComponent(self, input_text_box_metadata, cli=b1shim_cli)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 2",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_sar_eff(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ map",
                "name": "b1",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Input VOP file",
                "name": "vop",
                "button_function": "select_file",
                "required": True
            },
            {
                "button_label": "SAR factor",
                "name": "sar_factor",
                "default_text": "1.5",
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]

        component = InputComponent(self, input_text_box_metadata, cli=b1shim_cli)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 3",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer

    def create_sizer_phase_only(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "b1_shim_output")
        input_text_box_metadata = [
            {
                "button_label": "Input B1+ maps",
                "name": "b1",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, cli=b1shim_cli)
        run_component = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_b1shim --algo 4",
            output_paths=['TB1map_shimmed.nii.gz']
        )
        sizer = run_component.sizer
        return sizer
