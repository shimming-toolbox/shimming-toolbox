#!/usr/bin/python3
# -*- coding: utf-8 -*

import os
import webbrowser
import wx

from fsleyes_plugin_shimming_toolbox import __DIR_ST_PLUGIN_IMG__
from fsleyes_plugin_shimming_toolbox.components.input_component import InputComponent


class Tab(wx.ScrolledWindow):
    def __init__(self, parent, title, description):
        super().__init__(parent)
        self.title = title
        self.sizer_info = InfoSection(self, description).sizer
        self.terminal_component = parent.terminal_component
        self.SetScrollbars(1, 4, 1, 1)

    def create_sizer(self):
        """Create the parent sizer for the tab.

        Tab is divided into 2 main sizers:
            sizer_info | sizer_run
        """
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sizer_info, 0)
        sizer.AddSpacer(20)
        sizer.Add(self.sizer_run, 2)
        return sizer

    def create_sizer_run(self):
        """Create the run sizer containing tab-specific functionality."""
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.SetMinSize(400, 300)
        sizer.AddSpacer(10)
        return sizer

    def create_empty_component(self):
        component = InputComponent(panel=self, input_text_box_metadata=[])
        return component


class InfoSection:
    def __init__(self, panel, description):
        self.panel = panel
        self.description = description
        self.sizer = self.create_sizer()

    def create_sizer(self):
        """Create the left sizer containing generic Shimming Toolbox information."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Load ShimmingToolbox logo saved as a png image, rescale it, and return it as a wx.Bitmap image.
        st_logo = wx.Image(os.path.join(__DIR_ST_PLUGIN_IMG__, 'shimming_toolbox_logo.png'), wx.BITMAP_TYPE_PNG)
        st_logo.Rescale(int(st_logo.GetWidth() * 0.2), int(st_logo.GetHeight() * 0.2), wx.IMAGE_QUALITY_HIGH)
        st_logo = st_logo.ConvertToBitmap()
        logo = wx.StaticBitmap(parent=self.panel, id=-1, bitmap=st_logo, pos=wx.DefaultPosition)
        width = logo.Size[0]
        sizer.Add(logo, flag=wx.SHAPED, proportion=1)
        sizer.AddSpacer(10)

        # Create a "Documentation" button that redirects towards https://shimming-toolbox.org/en/latest/
        rtd_logo = wx.Bitmap(os.path.join(__DIR_ST_PLUGIN_IMG__, 'RTD.png'), wx.BITMAP_TYPE_PNG)
        button_documentation = wx.Button(self.panel, label="Documentation")
        button_documentation.Bind(wx.EVT_BUTTON, self.open_documentation_url)
        button_documentation.SetBitmap(rtd_logo)
        sizer.Add(button_documentation, flag=wx.EXPAND)
        sizer.AddSpacer(10)

        # Add a short description of what the current tab does
        description_text = wx.StaticText(self.panel, id=-1, label=self.description)
        description_text.Wrap(width)
        sizer.Add(description_text)
        return sizer

    def open_documentation_url(self, event):
        """Redirect ``button_documentation`` to the ``shimming-toolbox`` page."""
        webbrowser.open('https://shimming-toolbox.org/en/latest/')
