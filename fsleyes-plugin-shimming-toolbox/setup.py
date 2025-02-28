from setuptools import setup, find_packages
from os import path

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

path_version = path.join(here, 'fsleyes_plugin_shimming_toolbox', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()

requirements_pinned_path = path.join(path.dirname(here), "requirements_stplugin-pinned.txt")
requirements_path = path.join(path.dirname(here), "requirements_stplugin.txt")
if path.exists(requirements_pinned_path):
    with open(requirements_pinned_path) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
elif path.exists(requirements_path):
    with open(requirements_path) as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    raise FileNotFoundError(f"requirements_stplugin.txt not found at {requirements_path}")

setup(
    name='fsleyes-plugin-shimming-toolbox',
    version=version,
    install_requires=install_requires,
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