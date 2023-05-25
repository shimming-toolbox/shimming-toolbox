import os
from pathlib import Path


HOME_DIR = str(Path.home())
__CURR_DIR__ = os.getcwd()
__ST_DIR__ = f"{HOME_DIR}/shimming-toolbox"
__DIR_ST_PLUGIN__ = os.path.dirname(os.path.realpath(__file__))
