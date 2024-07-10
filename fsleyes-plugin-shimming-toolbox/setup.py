from setuptools import setup, find_packages

setup(
    name='fsleyes-plugin-shimming-toolbox',
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
