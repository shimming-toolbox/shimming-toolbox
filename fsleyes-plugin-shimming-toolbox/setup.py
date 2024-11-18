from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

path_version = path.join(here, 'fsleyes_plugin_shimming_toolbox', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()

setup(
    name='fsleyes-plugin-shimming-toolbox',
    version=version,
    install_requires=[
        "imageio",
        'pre-commit>=2.10.0'
    ],
    packages=find_packages(exclude=['.git']),
    include_package_data=True,
    entry_points={
        'fsleyes_controls': [
            'Shimming Toolbox = fsleyes_plugin_shimming_toolbox.st_plugin:STControlPanel'
        ],
        'fsleyes_layouts': [
            'Shimming Toolbox = fsleyes_plugin_shimming_toolbox.st_plugin:STLayout'
        ],
    }
)
