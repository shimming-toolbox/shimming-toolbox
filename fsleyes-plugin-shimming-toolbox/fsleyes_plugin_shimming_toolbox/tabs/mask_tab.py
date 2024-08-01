#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__
from fsleyes_plugin_shimming_toolbox.tabs.tab import Tab
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.components.run_component import RunComponent

from shimmingtoolbox.cli.mask import box, rect, threshold, sphere, bet


class MaskTab(Tab):
    def __init__(self, parent, title="Mask"):

        self.run_component_sphere = None
        self.run_component_box = None
        self.run_component_rect = None
        self.run_component_thr = None
        self.choice_box = None

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
            },
            {
                "name": "Sphere",
                "sizer_function": self.create_sizer_sphere
            },
            {
                "name": "BET",
                "sizer_function": self.create_sizer_bet
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
            sizer = self.sizer_run.GetItem(position)
            sizer.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self, choices=self.dropdown_choices, name="mask_algorithms")
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_run.Add(self.choice_box)
        self.sizer_run.AddSpacer(10)

    def create_sizer_threshold(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "output_mask_threshold")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "required": True
            },
            {
                "button_label": "Threshold",
                "default_text": "30",
                "name": "thr",
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, cli=threshold)
        self.run_component_thr = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask threshold"
        )
        sizer = self.run_component_thr.sizer
        return sizer

    def create_sizer_rect(self):
        path_output = os.path.join(__CURR_DIR__, "output_mask_rect")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 2,
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 2,
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, cli=rect)
        self.run_component_rect = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask rect"
        )
        sizer = self.run_component_rect.sizer
        return sizer

    def create_sizer_box(self):
        path_output = os.path.join(__CURR_DIR__, "output_mask_box")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "required": True
            },
            {
                "button_label": "Size",
                "name": "size",
                "n_text_boxes": 3,
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 3,
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, box)
        self.run_component_box = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask box"
        )
        sizer = self.run_component_box.sizer
        return sizer

    def create_sizer_sphere(self):
        path_output = os.path.join(__CURR_DIR__, "output_mask_sphere")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "required": True
            },
            {
                "button_label": "Radius",
                "name": "radius",
                "required": True
            },
            {
                "button_label": "Center",
                "name": "center",
                "n_text_boxes": 3,
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, sphere)
        self.run_component_sphere = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask sphere"
        )
        sizer = self.run_component_sphere.sizer
        return sizer
    
    def create_sizer_bet(self):
        path_output = os.path.join(__CURR_DIR__, "output_mask_bet")
        input_text_box_metadata = [
            {
                "button_label": "Input",
                "button_function": "select_from_overlay",
                "name": "input",
                "required": True
            },
            {
                "button_label": "fractional intensity threshold",
                "name": "f_param",
                "default_text": "0.5",
                "required": True
            },
            {
                "button_label": "Vertical gradient",
                "name": "g_param",
                "default_text": "0.0",
                "required": True
            },
            {
                "button_label": "Output File",
                "button_function": "select_folder",
                "default_text": os.path.join(path_output, "mask.nii.gz"),
                "name": "output",
            }
        ]
        component = InputComponent(self, input_text_box_metadata, bet)
        self.run_component_sphere = RunComponent(
            panel=self,
            list_components=[component],
            st_function="st_mask bet"
        )
        sizer = self.run_component_sphere.sizer
        return sizer
