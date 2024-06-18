#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import wx

from fsleyes_plugin_shimming_toolbox import __CURR_DIR__
from fsleyes_plugin_shimming_toolbox.tabs.tab import Tab
from fsleyes_plugin_shimming_toolbox.components.dropdown_component import DropdownComponent
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.components.run_component import RunComponent
from fsleyes_plugin_shimming_toolbox.components.checkbox_component import CheckboxComponent

from shimmingtoolbox.cli.b0shim import dynamic as dynamic_cli
from shimmingtoolbox.cli.b0shim import realtime_dynamic as realtime_cli
from shimmingtoolbox.cli.b0shim import max_intensity as max_intensity_cli


class B0ShimTab(Tab):
    def __init__(self, parent, title="B0 Shim"):

        self.dropdown_coil_format_rt = None
        self.dropdown_slice_rt = None
        self.dropdown_opt_rt = None
        self.dropdown_scanner_order_rt = None
        self.dropdown_coil_format_dyn = None
        self.dropdown_slice_dyn = None
        self.dropdown_opt_dyn = None
        self.dropdown_scanner_order_dyn = None
        self.choice_box = None
        self.run_component_mi = None
        self.run_component_rt = None
        self.run_component_dyn = None

        description = "Perform B0 shimming.\n\n" \
                      "Select the shimming algorithm from the dropdown list."
        super().__init__(parent, title, description)

        self.sizer_run = self.create_sizer_run()
        self.positions = {}
        self.dropdown_metadata = [
            {
                "name": "Dynamic/volume",
                "sizer_function": self.create_sizer_dynamic_shim
            },
            {
                "name": "Realtime Dynamic",
                "sizer_function": self.create_sizer_realtime_shim
            },
            {
                "name": "Maximum Intensity",
                "sizer_function": self.create_sizer_max_intensity
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
        if self.choice_box.GetSelection() < 0:
            selection = self.choice_box.GetString(0)
            self.choice_box.SetSelection(0)
        else:
            selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            run_component_sizer_item = self.sizer_run.GetItem(self.positions[selection])
            run_component_sizer_item.Show(True)

            # When doing Show(True), we show everything in the sizer, we need to call the dropdowns that can contain
            # items to show the appropriate things according to their current choice.
            if selection == 'Dynamic/volume':
                self.dropdown_slice_dyn.on_choice(None)
                self.dropdown_coil_format_dyn.on_choice(None)
                self.checkbox_scanner_order_dyn.on_choice(None)
                self.dropdown_opt_dyn.on_choice(None)
            elif selection == 'Realtime Dynamic':
                self.dropdown_slice_rt.on_choice(None)
                self.dropdown_coil_format_rt.on_choice(None)
                self.checkbox_scanner_order_rt.on_choice(None)
                self.dropdown_opt_rt.on_choice(None)
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
        self.choice_box = wx.Choice(self, choices=self.dropdown_choices, name="b0shim_algorithms")
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        self.sizer_run.Add(self.choice_box)
        self.sizer_run.AddSpacer(10)

    def create_sizer_dynamic_shim(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "output_dynamic_shim")

        # no_arg is used here since a --coil option must be used for each of the coils (defined add_input_coil_boxes)
        input_text_box_metadata_coil = [
            {
                "button_label": "Number of Custom Coils",
                "button_function": "add_input_coil_boxes_dyn",
                "name": "no_arg_ncoils_dyn",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
            }
        ]
        self.component_coils_dyn = InputComponent(self, input_text_box_metadata_coil)

        input_text_box_metadata_inputs = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Anat",
                "name": "anat",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Mask Dilation Kernel Size",
                "name": "mask-dilation-kernel-size",
                "default_text": "3",
            }
        ]

        component_inputs = InputComponent(self, input_text_box_metadata_inputs, cli=dynamic_cli)

        input_text_box_metadata_slice = [
            {
                "button_label": "Slice Factor",
                "name": "slice-factor",
                "default_text": "1",
            },
        ]
        component_slice_int = InputComponent(self, input_text_box_metadata_slice, cli=dynamic_cli)
        component_slice_seq = InputComponent(self, input_text_box_metadata_slice, cli=dynamic_cli)

        output_metadata = [
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]
        component_output = InputComponent(self, output_metadata, cli=dynamic_cli)

        input_text_box_metadata_scanner = [
            {
                "button_label": "Scanner constraints",
                "button_function": "select_file",
                "name": "scanner-coil-constraints",
            },
        ]
        component_scanner = InputComponent(self, input_text_box_metadata_scanner, cli=dynamic_cli)

        dropdown_scanner_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Slicewise per Coil",
                "option_value": "slicewise-coil"
            },
            {
                "label": "Chronological per Channel",
                "option_value": "chronological-ch"
            },
            {
                "label": "Chronological per Coil",
                "option_value": "chronological-coil"
            },
            {
                "label": "Gradient per Channel",
                "option_value": "gradient"
            },
        ]

        dropdown_scanner_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_format_metadata,
            option_name='output-file-format-scanner',
            label="Scanner Output Format",
            cli=dynamic_cli
        )

        checkbox_scanner_order_metadata = [
            {
                "label": "f0",
                "option_value": "0"
            },
            {
                "label": "1",
                "option_value": "1"
            },
            {
                "label": "2",
                "option_value": "2"
            },
            {
                "label": "3",
                "option_value": "3"
            }
        ]

        self.checkbox_scanner_order_dyn = CheckboxComponent(
            panel=self,
            label="Scanner Order",
            checkbox_metadata=checkbox_scanner_order_metadata,
            option_name='scanner-coil-order',
            components_dict=[{'object': dropdown_scanner_format, 'checkbox': ['f0', '1', '2', '3']},
                             {'object': component_scanner, 'checkbox': ['f0', '1', '2', '3']}],
        )

        dropdown_ovf_metadata = [
            {
                "label": "delta",
                "option_value": "delta"
            },
            {
                "label": "absolute",
                "option_value": "absolute"
            }
        ]

        dropdown_ovf = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_ovf_metadata,
            label="Output Value Format",
            option_name='output-value-format',
            cli=dynamic_cli
        )

        reg_factor_metadata = [
            {
                "button_label": "Regularization factor",
                "default_text": '0.0',
                "name": 'regularization-factor',
            }
        ]
        component_reg_factor_lsq = InputComponent(self, reg_factor_metadata, cli=dynamic_cli)
        component_reg_factor_qp = InputComponent(self, reg_factor_metadata, cli=dynamic_cli)

        criteria_dropdown_metadata = [
            {
                "label": "Mean Squared Error",
                "option_value": "mse",
            },
            {
                "label": "Mean Absolute Error",
                "option_value": "mae",
            },
            {
                "label": "Root Mean Squared Error",
                "option_value": "rmse",
            }
        ]

        dropdown_crit = DropdownComponent(
            panel=self,
            dropdown_metadata=criteria_dropdown_metadata,
            label="Optimizer Criteria",
            option_name='optimizer-criteria',
            cli=dynamic_cli
        )

        dropdown_opt_metadata = [
            {
                "label": "Least Squares",
                "option_value": "least_squares"
            },
            {
                "label": "Pseudo Inverse",
                "option_value": "pseudo_inverse"
            },
            {
                "label": "Quad Prog",
                "option_value": "quad_prog"
            }
        ]

        self.dropdown_opt_dyn = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_opt_metadata,
            label="Optimizer",
            option_name='optimizer-method',
            list_components=[dropdown_crit, component_reg_factor_lsq,
                             self.create_empty_component(), component_reg_factor_qp],
            component_to_dropdown_choice=[0, 0, 1, 2],
            cli=dynamic_cli
        )

        dropdown_crit.add_dropdown_parent(self.dropdown_opt_dyn)

        dropdown_slice_metadata = [
            {
                "label": "Auto detect",
                "option_value": "auto"
            },
            {
                "label": "Sequential",
                "option_value": "sequential"
            },
            {
                "label": "Interleaved",
                "option_value": "interleaved"
            },
            {
                "label": "Volume",
                "option_value": "volume"
            },
        ]

        self.dropdown_slice_dyn = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_slice_metadata,
            label="Slice Ordering",
            cli=dynamic_cli,
            option_name='slices',
            list_components=[self.create_empty_component(),
                             component_slice_seq,
                             component_slice_int,
                             self.create_empty_component()]
        )

        dropdown_coil_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Slicewise per Coil",
                "option_value": "slicewise-coil"
            },
            {
                "label": "Chronological per Channel",
                "option_value": "chronological-ch"
            },
            {
                "label": "Chronological per Coil",
                "option_value": "chronological-coil"
            },
        ]

        dropdown_fatsat_metadata = [
            {
                "label": "Auto detect",
                "option_value": "auto"
            },
            {
                "label": "Yes",
                "option_value": "yes"
            },
            {
                "label": "No",
                "option_value": "no"
            },
        ]

        dropdown_fatsat1 = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_fatsat_metadata,
            option_name='fatsat',
            label="Fat Saturation",
            cli=dynamic_cli
        )

        dropdown_fatsat2 = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_fatsat_metadata,
            option_name='fatsat',
            label="Fat Saturation",
            cli=dynamic_cli
        )

        self.dropdown_coil_format_dyn = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_coil_format_metadata,
            label="Custom Coil Output Format",
            option_name='output-file-format-coil',
            cli=dynamic_cli,
            list_components=[self.create_empty_component(),
                             self.create_empty_component(),
                             dropdown_fatsat1,
                             dropdown_fatsat2]
        )

        dropdown_fatsat1.add_dropdown_parent(self.dropdown_coil_format_dyn)
        dropdown_fatsat2.add_dropdown_parent(self.dropdown_coil_format_dyn)

        self.run_component_dyn = RunComponent(
            panel=self,
            list_components=[self.component_coils_dyn, component_inputs, self.dropdown_opt_dyn, self.dropdown_slice_dyn,
                             self.checkbox_scanner_order_dyn,
                             self.dropdown_coil_format_dyn, dropdown_ovf, component_output],
            st_function="st_b0shim dynamic",
            output_paths=["fieldmap_calculated_shim_masked.nii.gz",
                          "fieldmap_calculated_shim.nii.gz"]
        )
        sizer = self.run_component_dyn.sizer
        return sizer

    def create_sizer_realtime_shim(self, metadata=None):
        path_output = os.path.join(__CURR_DIR__, "output_realtime_shim")

        # no_arg is used here since a --coil option must be used for each of the coils (defined add_input_coil_boxes)
        input_text_box_metadata_coil = [
            {
                "button_label": "Number of Custom Coils",
                "button_function": "add_input_coil_boxes_rt",
                "name": "no_arg_ncoils_rt",
                "info_text": "Number of phase NIfTI files to be used. Must be an integer > 0.",
            }
        ]
        self.component_coils_rt = InputComponent(self, input_text_box_metadata_coil, cli=realtime_cli)

        input_text_box_metadata_inputs = [
            {
                "button_label": "Input Fieldmap",
                "name": "fmap",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Anat",
                "name": "anat",
                "button_function": "select_from_overlay",
                "required": True
            },
            {
                "button_label": "Input Respiratory Trace",
                "name": "resp",
                "button_function": "select_file",
                "required": True
            },
            {
                "button_label": "Input Mask Static",
                "name": "mask-static",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Input Mask Realtime",
                "name": "mask-riro",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Mask Dilation Kernel Size",
                "name": "mask-dilation-kernel-size",
                "default_text": "3",
            }
        ]

        component_inputs = InputComponent(self, input_text_box_metadata_inputs, cli=realtime_cli)

        input_text_box_metadata_slice = [
            {
                "button_label": "Slice Factor",
                "name": "slice-factor",
                "default_text": "1",
            },
        ]
        component_slice_int = InputComponent(self, input_text_box_metadata_slice, cli=realtime_cli)
        component_slice_seq = InputComponent(self, input_text_box_metadata_slice, cli=realtime_cli)

        output_metadata = [
            {
                "button_label": "Output Folder",
                "button_function": "select_folder",
                "default_text": path_output,
                "name": "output",
            }
        ]
        component_output = InputComponent(self, output_metadata, cli=realtime_cli)

        dropdown_scanner_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Chronological per Channel",
                "option_value": "chronological-ch"
            },
            {
                "label": "Gradient per Channel",
                "option_value": "gradient"
            },
        ]

        dropdown_scanner_format = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_scanner_format_metadata,
            label="Scanner Output Format",
            option_name='output-file-format-scanner',
            cli=realtime_cli
        )

        checkbox_scanner_order_metadata = [
            {
                "label": "f0",
                "option_value": "0"
            },
            {
                "label": "1",
                "option_value": "1"
            },
            {
                "label": "2",
                "option_value": "2"
            },
            {
                "label": "3",
                "option_value": "3"
            },

        ]

        input_text_box_metadata_scanner = [
            {
                "button_label": "Scanner constraints",
                "button_function": "select_file",
                "name": "scanner-coil-constraints",
            },
        ]
        component_scanner = InputComponent(self, input_text_box_metadata_scanner, cli=realtime_cli)

        self.checkbox_scanner_order_rt = CheckboxComponent(
            panel=self,
            label="Scanner Order",
            checkbox_metadata=checkbox_scanner_order_metadata,
            option_name='scanner-coil-order',
            components_dict=[{'object': dropdown_scanner_format, 'checkbox': ['f0', '1', '2', '3']},
                             {'object': component_scanner, 'checkbox': ['f0', '1', '2', '3']}],
            additional_sizer_dict={'info text': None, 'label': 'Scanner Order RIRO',
                                   'option name': 'scanner-coil-order-riro',
                                   'checkbox metadata': checkbox_scanner_order_metadata}
        )

        dropdown_ovf_metadata = [
            {
                "label": "delta",
                "option_value": "delta"
            },
            {
                "label": "absolute",
                "option_value": "absolute"
            }
        ]

        dropdown_ovf = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_ovf_metadata,
            label="Output Value Format",
            option_name='output-value-format',
            cli=realtime_cli
        )

        reg_factor_metadata = [
            {
                "button_label": "Regularization factor",
                "default_text": '0.0',
                "name": 'regularization-factor',
            }
        ]
        component_reg_factor_lsq = InputComponent(self, reg_factor_metadata, cli=dynamic_cli)
        component_reg_factor_qp = InputComponent(self, reg_factor_metadata, cli=dynamic_cli)

        criteria_dropdown_metadata = [
            {
                "label": "Mean Squared Error",
                "option_value": "mse",
            },
            {
                "label": "Mean Absolute Error",
                "option_value": "mae",
            },
            {
                "label": "Root Mean Squared Error",
                "option_value": "rmse",
            }
        ]

        dropdown_crit = DropdownComponent(
            panel=self,
            dropdown_metadata=criteria_dropdown_metadata,
            label="Optimizer Criteria",
            option_name='optimizer-criteria',
            cli=dynamic_cli
        )

        dropdown_opt_metadata = [
            {
                "label": "Least Squares",
                "option_value": "least_squares"
            },
            {
                "label": "Pseudo Inverse",
                "option_value": "pseudo_inverse"
            },
            {
                "label": "Quad Prog",
                "option_value": "quad_prog"
            }
        ]

        self.dropdown_opt_rt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_opt_metadata,
            label="Optimizer",
            option_name='optimizer-method',
            list_components=[dropdown_crit, component_reg_factor_lsq,
                             self.create_empty_component(), component_reg_factor_qp],
            component_to_dropdown_choice=[0, 0, 1, 2],
            cli=realtime_cli
        )

        dropdown_crit.add_dropdown_parent(self.dropdown_opt_rt)

        dropdown_slice_metadata = [
            {
                "label": "Auto detect",
                "option_value": "auto"
            },
            {
                "label": "Sequential",
                "option_value": "sequential"
            },
            {
                "label": "Interleaved",
                "option_value": "interleaved"
            },
            {
                "label": "Volume",
                "option_value": "volume"
            },
        ]

        self.dropdown_slice_rt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_slice_metadata,
            label="Slice Ordering",
            option_name='slices',
            list_components=[self.create_empty_component(),
                             component_slice_seq,
                             component_slice_int,
                             self.create_empty_component()],
            cli=realtime_cli
        )

        dropdown_fatsat_metadata = [
            {
                "label": "Auto detect",
                "option_value": "auto"
            },
            {
                "label": "Yes",
                "option_value": "yes"
            },
            {
                "label": "No",
                "option_value": "no"
            },
        ]

        dropdown_fatsat = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_fatsat_metadata,
            label="Fat Saturation",
            option_name='fatsat',
            cli=realtime_cli
        )

        dropdown_coil_format_metadata = [
            {
                "label": "Slicewise per Channel",
                "option_value": "slicewise-ch"
            },
            {
                "label": "Chronological per Channel",
                "option_value": "chronological-ch"
            }
        ]

        self.dropdown_coil_format_rt = DropdownComponent(
            panel=self,
            dropdown_metadata=dropdown_coil_format_metadata,
            label="Custom Coil Output Format",
            option_name='output-file-format-coil',
            cli=realtime_cli,
            list_components=[self.create_empty_component(),
                             dropdown_fatsat]
        )

        dropdown_fatsat.add_dropdown_parent(self.dropdown_coil_format_rt)

        self.run_component_rt = RunComponent(
            panel=self,
            list_components=[self.component_coils_rt, component_inputs, self.dropdown_opt_rt,
                             self.dropdown_slice_rt,
                             self.checkbox_scanner_order_rt,
                             self.dropdown_coil_format_rt, dropdown_ovf, component_output],
            st_function="st_b0shim realtime-dynamic",
            # TODO: output paths
            output_paths=[]
        )
        sizer = self.run_component_rt.sizer
        return sizer

    def create_sizer_max_intensity(self, metadata=None):
        fname_output = os.path.join(__CURR_DIR__, "output_maximum_intensity", "shim_index.txt")

        inputs_metadata = [
            {
                "button_label": "Input File",
                "button_function": "select_from_overlay",
                "required": True,
                "name": "input",
            },
            {
                "button_label": "Input Mask",
                "name": "mask",
                "button_function": "select_from_overlay",
            },
            {
                "button_label": "Output File",
                "default_text": fname_output,
                "name": "output",
            }
        ]
        component_inputs = InputComponent(self, inputs_metadata, cli=max_intensity_cli)

        self.run_component_mi = RunComponent(
            panel=self,
            list_components=[component_inputs],
            st_function="st_b0shim max-intensity",
            output_paths=[]
        )
        sizer = self.run_component_mi.sizer
        return sizer
