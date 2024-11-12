#!/usr/bin/python3
# -*- coding: utf-8 -*

import wx

from fsleyes_plugin_shimming_toolbox.components.component import Component, get_help_text
from fsleyes_plugin_shimming_toolbox.components.input_component import get_command_dict
from fsleyes_plugin_shimming_toolbox.text_with_button import create_info_icon


class DropdownComponent(Component):
    def __init__(self, panel, dropdown_metadata, label, option_name, list_components=[], info_text=None, cli=None,
                 component_to_dropdown_choice=None):
        """ Create a dropdown list

        Args:
            panel (wx.Panel): A panel is a window on which controls are placed.
            dropdown_metadata (list)(dict): A list of dictionaries where the dictionaries have the
                required keys: ``label``, ``option_value``.
                .. code::

                    {
                        "label": The label for the dropdown box
                        "option_value": The value linked to the option in the CLI
                    }

            label (str): Label of the button describing the dropdown
            option_name (str): Name of the options of the dropdown, start with 'no_arg' if it is not an option
            list_components (list): list of Components
            info_text (str): Info message displayed when hovering over the "i" icon. Leave blank to auto fill using option_name
            cli (function): CLI function used by the dropdown
            component_to_dropdown_choice (list): Tells which component associates with which dropdown selection.
                                                 If None, assumes 1:1.
        """
        super().__init__(panel, list_components)

        self.choice_box_sizer = None
        self.choice_box = None

        self.dropdown_metadata = dropdown_metadata
        self.label = label
        self.info_text = info_text
        self.positions = {}
        self.input_text_boxes = {}

        # If there is a dropdown parent, it needs to be added after instanciantion with self.add_dropdown_parent(parent)
        self.dropdown_parent = None

        self.dropdown_children = self.get_dropdown_children()
        self.sizer = self.create_sizer()
        self.dropdown_choices = [item["label"] for item in self.dropdown_metadata]
        self.option_name = option_name
        self.cli = cli

        if component_to_dropdown_choice is None:
            self.component_to_dropdown_choice = range(len(self.list_components))
        else:
            self.component_to_dropdown_choice = component_to_dropdown_choice

        self.add_text_info()
        self.create_choice_box()
        self.create_dropdown_sizers()
        self.on_choice(None)

    def get_dropdown_children(self):
        """ Finds the DropdownComponent from list_components and adds it to the list of children (dropdown_children)"""
        dropdown_children = []
        for component in self.list_components:
            if type(component) == DropdownComponent:
                dropdown_children.append(component)
        return dropdown_children

    def add_dropdown_parent(self, parent):
        """ Method to add the parent dropdown after instanciation of the class"""
        self.dropdown_parent = parent

    def add_text_info(self):
        """ This function adds the help text. """

        if self.info_text is None:
            if self.cli is not None:
                description = get_help_text(self.cli, self.option_name)
                self.info_text = description
            else:
                self.info_text = ""

    def create_dropdown_sizers(self):
        for index in range(len(self.list_components)):
            sizer = self.list_components[index].sizer
            self.sizer.Add(sizer, 0, wx.EXPAND)

            # Map list index to the dropdown selection
            dd_index = self.component_to_dropdown_choice[index]
            if self.dropdown_choices[dd_index] not in self.positions.keys():
                self.positions[self.dropdown_choices[dd_index]] = []
            self.positions[self.dropdown_choices[dd_index]].append(self.sizer.GetItemCount() - 1)

    def unshow_choice_box_sizers(self):
        """Set the Show variable to false for all sizers of the choice box widget"""
        for list_position in self.positions.values():
            for position in list_position:
                sizer = self.sizer.GetItem(position)
                sizer.Show(False)

    def create_choice_box(self):
        self.choice_box = wx.Choice(self.panel, choices=self.dropdown_choices, name=self.option_name)
        self.choice_box.Bind(wx.EVT_CHOICE, self.on_choice)
        button = wx.Button(self.panel, -1, label=self.label)
        self.choice_box_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.choice_box_sizer.Add(create_info_icon(self.panel, self.info_text), 0, wx.ALIGN_LEFT | wx.RIGHT, 7)
        self.choice_box_sizer.Add(button, 0, wx.ALIGN_LEFT | wx.RIGHT, 10)
        self.choice_box_sizer.Add(self.choice_box)
        self.sizer.Add(self.choice_box_sizer)
        self.sizer.AddSpacer(10)

    def on_choice(self, event, is_propagating_up=True):
        """ To enable nested dropdowns, every time a user selects a dropdown, we need to recalculate all other
            dropdowns and associated options. Dropdowns affect other dropdowns.

            is_propagating_up tells the dropdown to propagate the on_choice() command to the parent drop down until
            there is no more parent. Moreover, on each call of on_choice(), the propagation will also go down. This
            allows to recalculate each dropdown. There is some redundancy since we would really just want the most
            parent dropdown to send the down propagation, but this will do for now.
        """
        # Get the selection from the choice box widget
        if self.choice_box.GetSelection() < 0:
            selection = self.choice_box.GetString(0)
            self.choice_box.SetSelection(0)
        else:
            selection = self.choice_box.GetString(self.choice_box.GetSelection())

        # Unshow everything then show the correct item according to the choice box
        self.unshow_choice_box_sizers()
        if selection in self.positions.keys():
            for a_index in self.positions[selection]:
                sizer_item = self.sizer.GetItem(a_index)
                sizer_item.Show(True)
        else:
            pass

        index_dd = self.find_index(selection)
        if selection in self.positions.keys():
            # Add the sizers to the current list of options
            # find indexes of list_components that are associated with index dropdown
            indexes = [i for i, e in enumerate(self.component_to_dropdown_choice) if e == index_dd]
            self.input_text_boxes = {}
            for index_comp in indexes:
                component = self.list_components[index_comp]
                # Merge both input_text_boxes
                self.input_text_boxes.update(component.input_text_boxes)
                # Propagate down to show or hide the relevant sub DropdownComponents
                if type(component) == DropdownComponent:
                    component.on_choice(None, is_propagating_up=False)

        # Add the dropdown to the list of options
        self.input_text_boxes[self.option_name] = [self.dropdown_metadata[index_dd]["option_value"]]

        # Update the parent if there is one (This allows to propagate nested dropdown selection)
        if is_propagating_up:
            if self.dropdown_parent is not None:
                self.dropdown_parent.on_choice(None)

        # Update the window
        self.panel.SetVirtualSize(self.panel.sizer_run.GetMinSize())
        self.panel.Layout()

    def find_index(self, label):
        for index in range(len(self.dropdown_metadata)):
            if self.dropdown_metadata[index]["label"] == label:
                return index

        # Return index 0 if it is not found
        return 0

    def create_sizer(self):
        """Create a sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer
    
    def get_command(self):
        """Return the selcted options in the dropdown"""
        command = []
        output = None
        load_in_overlay = []
        for name, input_text_box_list in self.input_text_boxes.items():
                if name.startswith('no_arg'):
                    continue

                for input_text_box in input_text_box_list:
                    # Allows to choose from a dropdown
                    if type(input_text_box) == str:
                        command.extend(['--'+ name, input_text_box])
                    else:
                        input_text_boxes = {name: input_text_box_list}
                        
                        cmd, output, load_in_overlay = get_command_dict(input_text_boxes)
                        command.extend(cmd)
        return command, output, load_in_overlay
