#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wx

result_event_type = wx.NewEventType()
EVT_RESULT = wx.PyEventBinder(result_event_type, 1)

log_event_type = wx.NewEventType()
EVT_LOG = wx.PyEventBinder(log_event_type, 1)


class ResultEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id, name):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.data = ""
        self.name = name

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data


class LogEvent(wx.PyCommandEvent):
    def __init__(self, evtType, id, name):
        wx.PyCommandEvent.__init__(self, evtType, id)
        self.data = ""
        self.name = name

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data