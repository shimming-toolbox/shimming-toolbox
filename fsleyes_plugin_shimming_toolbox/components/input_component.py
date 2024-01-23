#!/usr/bin/python3
# -*- coding: utf-8 -*

import wx

from fsleyes_plugin_shimming_toolbox.components.component import Component, get_help_text, RunArgumentErrorST
from fsleyes_plugin_shimming_toolbox.text_with_button import TextWithButton


class InputComponent(Component):
    """ Define cli to automatically generate help text """
    def __init__(self, panel, input_text_box_metadata, cli=None):
        super().__init__(panel)
        self.sizer = self.create_sizer()
        self.input_text_boxes = {}
        self.input_text_box_metadata = input_text_box_metadata
        self.cli = cli
        self.add_text_info()
        self.add_input_text_boxes()

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer

    def add_text_info(self):
        """ This function adds the help text to the metadata. This function
            needs to be called before creating the buttons.

        """
        for i, twb_dict in enumerate(self.input_text_box_metadata):
            if not ('info_text' in twb_dict.keys()) and (self.cli is not None) and ('name' in twb_dict.keys()):
                description = get_help_text(self.cli, twb_dict['name'])
                self.input_text_box_metadata[i]['info_text'] = description

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
                required=twb_dict.get("required", False),
                load_in_overlay=twb_dict.get("load_in_overlay", False)
            )
            self.add_input_text_box(text_with_button, twb_dict.get("name", "default"))

    def add_input_text_box(self, text_with_button, name, spacer_size=10):
        box = text_with_button.create()
        self.sizer.Add(box, 1, wx.EXPAND)
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
        pass

    def get_command(self):
        """ Returns the arguments of the input text boxes.

        Returns:
            args (string): argmuents of the input text boxes
        """

        return get_command_dict(self.input_text_boxes)


def get_command_dict(input_text_boxes):

    command = []
    command_list_arguments = []
    command_list_options = []
    output = ""
    load_in_overlay = []
    for name, input_text_box_list in input_text_boxes.items():

        if name.startswith('no_arg'):
            continue

        for input_text_box in input_text_box_list:
            is_arg = False
            option_values = []
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
                        is_arg = True
                    # Normal options
                    else:
                        if name == "output":
                            output = arg
                        elif input_text_box.load_in_overlay:
                            load_in_overlay.append(arg)
                        
                        option_values.append(arg)
                        
                # If its an argument don't include it as an option, if the option list is empty don't either
            if not is_arg and option_values:
                command_list_options.append((name, option_values))

    # Arguments don't need "-"
    for arg in command_list_arguments:
        command.append(arg)

    # Handles options
    for name, args in command_list_options:
        command.append('--' + name)
        for arg in args:
            command.append(arg)

    return command, output, load_in_overlay
    