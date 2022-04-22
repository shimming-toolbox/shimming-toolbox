#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import subprocess
from threading import Thread
import wx

from fsleyes_plugin_shimming_toolbox.events import result_event_type, EVT_RESULT, ResultEvent
from fsleyes_plugin_shimming_toolbox.events import log_event_type, EVT_LOG, LogEvent

HOME_DIR = str(Path.home())
PATH_ST_VENV = f"{HOME_DIR}/shimming-toolbox/python/envs/st_venv/bin"


class WorkerThread(Thread):
    def __init__(self, notify_window, cmd, name):
        Thread.__init__(self)
        self._notify_window = notify_window
        self.cmd = cmd
        self.name = name
        self.start()

    def run(self):

        try:
            env = os.environ.copy()
            # It seems to default to the Python executalble instead of the Shebang, removing it fixes it
            env["PYTHONEXECUTABLE"] = ""
            env["PATH"] = PATH_ST_VENV + ":" + env["PATH"]
            
            # Run command using realtime output
            process = subprocess.Popen(self.cmd.split(' '),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True,
                                       env=env)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    evt = LogEvent(log_event_type, -1, self.name)
                    evt.set_data(output.strip())
                    wx.PostEvent(self._notify_window, evt)

            rc = process.poll()
            evt = ResultEvent(result_event_type, -1, self.name)
            evt.set_data(rc)
            wx.PostEvent(self._notify_window, evt)
        except Exception as err:
            # Send the error if there was one
            evt = ResultEvent(result_event_type, - 1, self.name)
            evt.set_data(err)
            wx.PostEvent(self._notify_window, evt)
