import wx
from fsleyes_plugin_shimming_toolbox.components.component import Component
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent
from fsleyes_plugin_shimming_toolbox.text_with_button import create_info_icon


class CheckboxComponent(Component):
    def __init__(self, panel, label, checkbox_metadata, option_name, components_dict={}, info_text=None, additional_sizer_dict=None):
        """
        Create a checkbox object
        Args:
            panel (wx.Panel): A panel is a window on which controls are placed.
            label (str): Label of the button describing the checkbox
            checkbox_metadata (list)(dict): A list of dictionaries where the dictionaries have the
                required keys: ``label``, ``option_value``.
                .. code::
                    
                        {
                            "label": The label for the checkbox
                            "option_value": The value linked to the option in the CLI
                        }
            option_name (str): Name of the options of the checkbox, start with 'no_arg' if it is not an option
            components_dict (dict): Dictionary of components to shown when the checkbox is checked
            info_text (str): Info message displayed when hovering over the "i" icon. Leave blank to auto fill using option_name
            additional_sizer_dict (dict): Dictionary of additional checkbox to be added to the sizer
        """
        self.sizer = self.create_sizer()

        self.checkbox_metadata = checkbox_metadata
        self.option_name = option_name
        self.checkboxes = []
        self.checkboxes_riro = []
        
        self.panel = panel
        self.label = label
        self.info_text = info_text
        self.children = components_dict
        self.additional_sizer_dict = additional_sizer_dict
        
        self.create_display()

    def create_display(self):
        # Button
        self.info_icon = create_info_icon(self.panel, self.info_text)
        self.button = wx.Button(self.panel, -1, label=self.label)

        self.add_checkbox_sizer(self.checkbox_metadata, self.info_icon, self.button)
        
        if self.additional_sizer_dict is not None:
            info_icon = create_info_icon(self.panel, self.additional_sizer_dict["info text"])
            button = wx.Button(self.panel, -1, label=self.additional_sizer_dict["label"])
            self.add_checkbox_sizer(self.additional_sizer_dict["checkbox metadata"], 
                                    info_icon, button, riro=True)

        # Add children
        self.add_children()

        wx.CallAfter(self.on_choice, None)

    def add_checkbox_sizer(self, checkbox_metadata, info_icon, button, riro=False):
        temp_checkboxes = []
        # Checkboxes
        for metadata in checkbox_metadata:
            temp_checkboxes.append(wx.CheckBox(self.panel, label=metadata["label"]))

        # Add checkboxes to sizer
        checkbox_sizer = wx.BoxSizer(wx.HORIZONTAL)
        checkbox_sizer.Add(info_icon, 0, wx.ALIGN_LEFT | wx.RIGHT, 7)
        checkbox_sizer.Add(button, 0, wx.ALIGN_LEFT | wx.RIGHT, 10)
        for checkbox in temp_checkboxes:
            # Bind
            checkbox.Bind(wx.EVT_CHECKBOX, self.on_choice)
            checkbox_sizer.Add(checkbox, 0, wx.ALL, 5)        # Add to sizer + spacer below
        if riro:
            self.checkboxes_riro.extend(temp_checkboxes)
        else:
            self.checkboxes.extend(temp_checkboxes)
        self.sizer.Add(checkbox_sizer)
        self.sizer.AddSpacer(10)
    
    def add_children(self):
        for child in self.children:
            self.sizer.Add(child['object'].sizer, 0, wx.EXPAND)

    def on_choice(self, event):
        childrens_to_show = self.get_children_to_show()
        for child in self.children:
            if child['object'] in childrens_to_show:
                child['object'].sizer.ShowItems(True)
            else:
                child['object'].sizer.ShowItems(False)
        self.panel.SetVirtualSize(self.panel.sizer_run.GetMinSize())
        self.panel.Layout()

    def get_children_to_show(self):
        """Get the children to show based on the checkbox selection"""
        checked_indices = {checkbox.GetLabel() for checkbox in self.checkboxes + self.checkboxes_riro if checkbox.GetValue()}
        children_to_show = [child['object'] for child in self.children if set(child['checkbox']).intersection(checked_indices)]

        return children_to_show

    def create_sizer(self):
        """Create the centre sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        return sizer

    def get_argument(self, checkboxes):
        args = ""
        option_values = {item['label']: item['option_value'] for item in self.checkbox_metadata}
        for checkbox in checkboxes:
            if checkbox.GetValue():
                label = checkbox.GetLabel()
                args += option_values[label] + ','
        if args == "":
            return "-1"
        
        return args[:-1]
    
    def get_command(self):
        command = []
        output = None
        overlay = []
        
        args = self.get_argument(self.checkboxes)

        if self.option_name == "arg":
            command.append(args)
        else:
            command.extend(['--' + self.option_name, args])

        if self.checkboxes_riro:
            args_riro = self.get_argument(self.checkboxes_riro)
            command.extend(['--' + self.additional_sizer_dict["option name"], args_riro])
            
        children_to_return = self.get_children_to_show()
        for child in children_to_return:
            if type(child) == InputComponent:
                cmd, output, overlay = child.get_command()
            else:
                cmd, _, _ = child.get_command()

            command.extend(cmd)

        return command, output, overlay
