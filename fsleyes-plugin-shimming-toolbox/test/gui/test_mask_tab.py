#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import nibabel as nib
import numpy as np
import os
import pathlib
from shimmingtoolbox.masking.shapes import shapes
import shutil
import tempfile
import time
import wx

from .test_tabs import get_notebook, set_notebook_page, get_tab, get_all_children, set_dropdown_selection, set_checkbox
from .. import realYield, run_with_orthopanel
from fsleyes_plugin_shimming_toolbox import __dir_testing__
from fsleyes_plugin_shimming_toolbox.tabs.mask_tab import MaskTab


def test_st_plugin_mask_bet():
    options = {
        'f_param': '1',
        'g_param': '0.2',
    }
    
    def _test_st_plugin_mask_bet(view, overlayList, displayCtx, options=options):
        __test_st_plugin_mask_bet(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_mask_bet)
    
def test_st_plugin_mask_erode():
    options = {
        'operation': 'Erode',
        'shape': 'Cube',
        'size': '3',
    }
    
    def _test_st_plugin_mask_modify(view, overlayList, displayCtx, options=options):
        __test_st_plugin_mask_modify(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_mask_modify)


def test_st_plugin_mask_dilate():
    options = {
        'operation': 'Dilate',
        'shape': 'Sphere',
        'size': '3',
    }
    
    def _test_st_plugin_mask_modify(view, overlayList, displayCtx, options=options):
        __test_st_plugin_mask_modify(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_mask_modify)
    
    
def __test_st_plugin_mask_bet(view, overlayList, displayCtx, options):
    """
    Test the Mask tab with the BET option.
    """
    nb_terminal = get_notebook(view)
    
    # Select the mask tab
    assert set_notebook_page(nb_terminal, "Mask")
    
    # Get the mask tab
    mask_tab = get_tab(nb_terminal, MaskTab)
    assert mask_tab is not None
    
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # fname for anat
        fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
        nii_anat = nib.load(fname_anat)
        
        # Fill the widgets with the Mask tab options
        list_widgets = []
        get_all_children(mask_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'mask_algorithms':
                    assert set_dropdown_selection(widget, 'BET')
        
        # Fill the widgets with the BET options
        list_widgets = []
        get_all_children(mask_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if widget.GetName() == 'f_param':
                widget.SetValue(options['f_param'])
            elif widget.GetName() == 'g_param':
                widget.SetValue(options['g_param'])
            elif widget.GetName() == 'input':
                widget.SetValue(fname_anat)
            elif widget.GetName() == 'output':
                widget.SetValue(os.path.join(tmp, 'mask_bet.nii.gz'))
                
        # Run the mask
        mask_tab.run_component_bet.run()
        
        # Search for the output for a maximum of 20 seconds
        for _ in range(20):
            realYield()
            ovrlay_file = overlayList.find("mask_bet_mask")
            time.sleep(1)
            if ovrlay_file:
                break
        
        # Make sure the output is correct
        assert ovrlay_file is not None
        assert os.path.exists(ovrlay_file.dataSource)
        

def __test_st_plugin_mask_modify(view, overlayList, displayCtx, options):
    """
    Test the Mask tab with the Erode option.
    """
    nb_terminal = get_notebook(view)
    
    # Select the mask tab
    assert set_notebook_page(nb_terminal, "Mask")
    
    # Get the mask tab
    mask_tab = get_tab(nb_terminal, MaskTab)
    assert mask_tab is not None
    
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        # fname for anat
        fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')
        nii_anat = nib.load(fname_anat)
        
        nx, ny, nz = nii_anat.shape
        mask = shapes(nii_anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)
        
        fname_output_file = "mask_modified"
        output_path = os.path.join(tmp, fname_output_file + ".nii.gz")
        # Fill the widgets with the Mask tab options
        list_widgets = []
        get_all_children(mask_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'mask_algorithms':
                    assert set_dropdown_selection(widget, 'Erode/Dilate')
        
        # Fill the widgets with the Erode options
        list_widgets = []
        get_all_children(mask_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if widget.GetName() == 'input':
                widget.SetValue(fname_anat)
            elif widget.GetName() == 'operation':
                assert set_dropdown_selection(widget, options['operation'])
            elif widget.GetName() == 'shape':
                assert set_dropdown_selection(widget, options['shape'])
            elif widget.GetName() == 'size':
                widget.SetValue(options['size'])
            elif widget.GetName() == 'output':
                widget.SetValue(output_path)
                
        # Run the mask
        mask_tab.run_component_modify.run()
        
        # Search for the output for a maximum of 20 seconds
        for _ in range(20):
            realYield()
            ovrlay_file = overlayList.find(fname_output_file)
            time.sleep(1)
            if ovrlay_file:
                break
        
        # Make sure the output is correct
        assert ovrlay_file is not None
        assert os.path.exists(ovrlay_file.dataSource)