import os
from pathlib import Path


HOME_DIR = str(Path.home())
__CURR_DIR__ = os.getcwd()
__ST_DIR__ = f"{HOME_DIR}/shimming-toolbox"
__DIR_ST_PLUGIN__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__DIR_ST_PLUGIN_IMG__ = os.path.join(__DIR_ST_PLUGIN__, 'fsleyes_plugin_shimming_toolbox', 'img')
__dir_testing__ = os.path.join(__DIR_ST_PLUGIN__, 'testing_data')
