#!/usr/bin/python3
# -*- coding: utf-8 -*

import abc


class Component:
    def __init__(self, panel, list_components=[]):
        self.panel = panel
        self.list_components = list_components

    @abc.abstractmethod
    def create_sizer(self):
        raise NotImplementedError

    # make sure that the create_sizer method has been implemented in the subclasses
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'create_sizer') and callable(subclass.create_sizer) or NotImplemented
    
    @abc.abstractmethod
    def get_command(self):
        raise NotImplementedError


def get_help_text(cli_function, name):
    """ Returns the help text of a cli function depending on its name. """
    for param in cli_function.params:
        # Try different versions of the input dashes
        for dashes in ['', '-', '--']:
            new_name = dashes + name
            if (new_name in param.opts) or new_name == param.human_readable_name:
                return param.help

    raise ValueError(f"Could not find param: {name} in {cli_function.name}")


class RunArgumentErrorST(Exception):
    """Exception for missing input arguments for CLI call."""
    pass
    