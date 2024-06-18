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
from fsleyes_plugin_shimming_toolbox.tabs.b0shim_tab import B0ShimTab


def test_st_plugin_b0shim_dyn_lsq_mse():
    options = {'optimizer-method': 'Least Squares',
               'optimizer-criteria': 'Mean Squared Error',
               'slices': 'Auto detect',
               'scanner-coil-order': 'f0',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'fatsat': 'Auto detect',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_dyn)


def test_st_plugin_b0shim_dyn_lsq_mae():
    options = {'optimizer-method': 'Least Squares',
               'optimizer-criteria': 'Mean Absolute Error',
               'slices': 'Auto detect',
               'scanner-coil-order': '1',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_dyn)


def test_st_plugin_b0shim_dyn_lsq_mse_coil_only():
    options = {'optimizer-method': 'Least Squares',
               'optimizer-criteria': 'Mean Absolute Error',
               'slices': 'Auto detect',
               'scanner-coil-order': '',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_dyn)


def test_st_plugin_b0shim_dyn_pi():
    options = {'optimizer-method': 'Pseudo Inverse',
               'slices': 'Auto detect',
               'scanner-coil-order': '1',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_dyn)


def test_st_plugin_b0shim_dyn_qp():
    options = {'optimizer-method': 'Quad Prog',
               'optimizer-criteria': 'Mean Squared Error',
               'slices': 'Auto detect',
               'scanner-coil-order': '1',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_dyn)


def __test_st_plugin_b0shim_dyn(view, overlayList, displayCtx, options):
    """ Makes sure the B0 shim tab runs (Add dummy input and simulate a click) """
    nb_terminal = get_notebook(view)

    # Select the b0Shim tab
    assert set_notebook_page(nb_terminal, 'B0 Shim')

    # Get the ST tab
    b0shim_tab = get_tab(nb_terminal, B0ShimTab)
    assert b0shim_tab is not None

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        nii_fmap, nii_anat, nii_mask, nii_coil, fm_data, anat_data, coil_data, _ = _define_inputs(fmap_dim=3)
        fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
        fname_fm_json = os.path.join(tmp, 'fmap.json')
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        fname_anat = os.path.join(tmp, 'anat.nii.gz')
        fname_anat_json = os.path.join(tmp, 'anat.json')
        fname_coil = os.path.join(tmp, 'coil.nii.gz')
        fname_coil_json = os.path.join(tmp, 'coil.json')

        _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                     nii_anat=nii_anat, fname_anat=fname_anat,
                     nii_mask=nii_mask, fname_mask=fname_mask,
                     nii_coil=nii_coil, fname_coil=fname_coil,
                     fm_data=fm_data, fname_fm_json=fname_fm_json,
                     anat_data=anat_data, fname_anat_json=fname_anat_json,
                     coil_data=coil_data, fname_coil_json=fname_coil_json)
        fname_output = os.path.join(tmp, 'shim')

        # Fill in the B0 shim tab options
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'b0shim_algorithms':
                    # Select the proper algorithm
                    assert set_dropdown_selection(widget, 'Dynamic/volume')

        # Select the dropdowns
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'optimizer-method':
                    assert set_dropdown_selection(widget, options['optimizer-method'])
                if widget.GetName() == 'slices':
                    assert set_dropdown_selection(widget, options['slices'])
                if widget.GetName() == 'output-value-format':
                    assert set_dropdown_selection(widget, options['output-value-format'])

        # Select the checkboxes
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.CheckBox) and widget.IsShown():
                if widget.Label in options['scanner-coil-order']:
                    assert set_checkbox(widget)

        # Select the dropdowns that are nested
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'optimizer-criteria':
                    assert set_dropdown_selection(widget, options['optimizer-criteria'])
                if widget.GetName() == 'output-file-format-scanner':
                    assert set_dropdown_selection(widget, options['output-file-format-scanner'])
                if widget.GetName() == 'output-file-format-scanner':
                    assert set_dropdown_selection(widget, options['output-file-format-coil'])
        # Select the dropdowns that are nested
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'fatsat':
                    assert set_dropdown_selection(widget, options['fatsat'])

        # Fill in the text boxes
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.TextCtrl) and widget.IsShown():
                if widget.GetName() == 'no_arg_ncoils_dyn':
                    widget.SetValue('1')
                    new_widget_list = []
                    counter = 0
                    get_all_children(b0shim_tab.sizer_run, new_widget_list)
                    for new_widget in new_widget_list:
                        if isinstance(new_widget, wx.TextCtrl) and new_widget.IsShown():
                            if new_widget.GetName() == 'input_coil_1' and counter == 0:
                                new_widget.SetValue(fname_coil)
                                counter += 1
                                realYield()
                            elif new_widget.GetName() == 'input_coil_1' and counter == 1:
                                new_widget.SetValue(fname_coil_json)
                                realYield()
                    realYield()
                if widget.GetName() == 'fmap':
                    widget.SetValue(fname_fmap)
                    realYield()
                if widget.GetName() == 'anat':
                    widget.SetValue(fname_anat)
                    realYield()
                if widget.GetName() == 'mask':
                    widget.SetValue(fname_mask)
                    realYield()
                if widget.GetName() == 'mask-dilation-kernel-size':
                    widget.SetValue('5')
                    realYield()
                if widget.GetName() == 'regularization-factor':
                    widget.SetValue('0.1')
                    realYield()
                if widget.GetName() == 'output':
                    widget.SetValue(fname_output)
                    realYield()

        # Call the function ran when clicking run button
        b0shim_tab.run_component_dyn.run()

        # Search for files in the overlay for a maximum of 20s
        time_limit = 20  # s
        for i in range(time_limit):
            realYield()
            overlay_file = overlayList.find("fieldmap_calculated_shim_masked")
            time.sleep(1)
            if overlay_file:
                break

        # Make sure there is an output in the overlay (that would mean the ST CLI ran)
        assert overlay_file is not None
        assert os.path.exists(fname_output)


def test_st_plugin_b0shim_rt_lsq_mse_coil():
    options = {'optimizer-method': 'Least Squares',
               'optimizer-criteria': 'Mean Squared Error',
               'slices': 'Volume',
               'scanner-coil-order': '1',
               'output-file-format-scanner': 'Slicewise per Channel',
               'output-file-format-coil': 'Slicewise per Channel',
               'output-value-format': 'delta'
               }

    def _test_st_plugin_b0shim_rt(view, overlayList, displayCtx, options=options):
        __test_st_plugin_b0shim_rt(view, overlayList, displayCtx, options=options)
    run_with_orthopanel(_test_st_plugin_b0shim_rt)


def __test_st_plugin_b0shim_rt(view, overlayList, displayCtx, options):
    """ Makes sure the B0 shim tab runs (Add dummy input and simulate a click) """
    nb_terminal = get_notebook(view)

    # Select the b0Shim tab
    assert set_notebook_page(nb_terminal, 'B0 Shim')

    # Get the ST tab
    b0shim_tab = get_tab(nb_terminal, B0ShimTab)
    assert b0shim_tab is not None

    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        nii_fmap, nii_anat, nii_mask, nii_coil, fm_data, anat_data, coil_data, fname_resp = _define_inputs(fmap_dim=4)
        fname_fmap = os.path.join(tmp, 'fmap.nii.gz')
        fname_fm_json = os.path.join(tmp, 'fmap.json')
        fname_mask = os.path.join(tmp, 'mask.nii.gz')
        fname_anat = os.path.join(tmp, 'anat.nii.gz')
        fname_anat_json = os.path.join(tmp, 'anat.json')
        fname_coil = os.path.join(tmp, 'coil.nii.gz')
        fname_coil_json = os.path.join(tmp, 'coil.json')
        fname_new_resp = os.path.join(tmp, 'respiration_data.resp')

        _save_inputs(nii_fmap=nii_fmap, fname_fmap=fname_fmap,
                     nii_anat=nii_anat, fname_anat=fname_anat,
                     nii_mask=nii_mask, fname_mask=fname_mask,
                     nii_coil=nii_coil, fname_coil=fname_coil,
                     fm_data=fm_data, fname_fm_json=fname_fm_json,
                     anat_data=anat_data, fname_anat_json=fname_anat_json,
                     coil_data=coil_data, fname_coil_json=fname_coil_json,
                     fname_resp=fname_resp, fname_new_resp=fname_new_resp)
        path_output = os.path.join(tmp, 'shim')

        # Fill in the B0 shim tab options
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'b0shim_algorithms':
                    # Select the proper algorithm
                    assert set_dropdown_selection(widget, 'Realtime Dynamic')

        # Select the dropdowns
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'optimizer-method':
                    assert set_dropdown_selection(widget, options['optimizer-method'])
                if widget.GetName() == 'slices':
                    assert set_dropdown_selection(widget, options['slices'])
                if widget.GetName() == 'output-value-format':
                    assert set_dropdown_selection(widget, options['output-value-format'])

        # Select the checkboxes
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.CheckBox) and widget.IsShown():
                if widget.Label in options['scanner-coil-order']:
                    assert set_checkbox(widget)

        # Select the dropdowns that are nested
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'optimizer-criteria':
                    assert set_dropdown_selection(widget, options['optimizer-criteria'])
                if widget.GetName() == 'output-file-format-scanner':
                    assert set_dropdown_selection(widget, options['output-file-format-scanner'])
                if widget.GetName() == 'output-file-format-scanner':
                    assert set_dropdown_selection(widget, options['output-file-format-coil'])
        # Select the dropdowns that are nested
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.Choice) and widget.IsShown():
                if widget.GetName() == 'fatsat':
                    assert set_dropdown_selection(widget, options['fatsat'])

        # Fill in the text boxes
        list_widgets = []
        get_all_children(b0shim_tab.sizer_run, list_widgets)
        for widget in list_widgets:
            if isinstance(widget, wx.TextCtrl) and widget.IsShown():
                if widget.GetName() == 'no_arg_ncoils_dyn':
                    widget.SetValue('1')
                    new_widget_list = []
                    counter = 0
                    get_all_children(b0shim_tab.sizer_run, new_widget_list)
                    for new_widget in new_widget_list:
                        if isinstance(new_widget, wx.TextCtrl) and new_widget.IsShown():
                            if new_widget.GetName() == 'input_coil_1' and counter == 0:
                                new_widget.SetValue(fname_coil)
                                counter += 1
                                realYield()
                            elif new_widget.GetName() == 'input_coil_1' and counter == 1:
                                new_widget.SetValue(fname_coil_json)
                                realYield()
                    realYield()
                if widget.GetName() == 'fmap':
                    widget.SetValue(fname_fmap)
                    realYield()
                if widget.GetName() == 'anat':
                    widget.SetValue(fname_anat)
                    realYield()
                if widget.GetName() == 'resp':
                    widget.SetValue(fname_new_resp)
                    realYield()
                if widget.GetName() == 'mask':
                    widget.SetValue(fname_mask)
                    realYield()
                if widget.GetName() == 'mask-dilation-kernel-size':
                    widget.SetValue('5')
                    realYield()
                if widget.GetName() == 'regularization-factor':
                    widget.SetValue('0.1')
                    realYield()
                if widget.GetName() == 'output':
                    widget.SetValue(path_output)
                    realYield()

        # Call the function ran when clicking run button
        b0shim_tab.run_component_rt.run()

        # Search for files in the overlay for a maximum of 20s
        time_limit = 20  # s
        for i in range(time_limit):
            realYield()

            found = os.path.exists(os.path.join(path_output, "fig_shimmed_vs_unshimmed.png"))

            # Todo once output is pushed to the overlay
            # overlay_file = overlayList.find("")
            # if overlay_file:
            #     found = True

            if found:
                break

            time.sleep(1)

        # Make sure there is an output in the overlay (that would mean the ST CLI ran)
        assert found
        assert os.path.exists(path_output)


def _define_inputs(fmap_dim):
    # fname for fmap
    fname_fmap = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.nii.gz')
    nii = nib.load(fname_fmap)

    fname_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_fieldmap.json')

    fm_data = json.load(open(fname_json))

    if fmap_dim == 4:
        nii_fmap = nii
        resp = os.path.join(__dir_testing__, 'ds_b0', 'derivatives', 'sub-realtime', 'sub-realtime_PMUresp_signal.resp')
    elif fmap_dim == 3:
        nii_fmap = nib.Nifti1Image(np.mean(nii.get_fdata(), axis=3), nii.affine, header=nii.header)
        resp = None
    elif fmap_dim == 2:
        nii_fmap = nib.Nifti1Image(nii.get_fdata()[..., 0, 0], nii.affine, header=nii.header)
        resp = None
    else:
        raise ValueError("Supported Dimensions are 2, 3 or 4")

    # fname for anat
    fname_anat = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.nii.gz')

    nii_anat = nib.load(fname_anat)

    fname_anat_json = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'anat', 'sub-realtime_unshimmed_e1.json')
    anat_data = json.load(open(fname_anat_json))
    anat_data['ScanOptions'] = ['FS']

    anat = nii_anat.get_fdata()

    # Set up mask: Cube
    # static
    nx, ny, nz = anat.shape
    mask = shapes(anat, 'cube',
                  center_dim1=int(nx / 2),
                  center_dim2=int(ny / 2),
                  len_dim1=10, len_dim2=10, len_dim3=nz - 10)

    nii_mask = nib.Nifti1Image(mask.astype(np.uint8), nii_anat.affine)

    fname_coil_nii = os.path.join(__dir_testing__, 'ds_coil', 'NP15ch_coil_profiles.nii.gz')
    nii_coil = nib.load(fname_coil_nii)
    fname_coil_json = os.path.join(__dir_testing__, 'ds_coil', 'NP15ch_config.json')
    coil_data = json.load(open(fname_coil_json))

    return nii_fmap, nii_anat, nii_mask, nii_coil, fm_data, anat_data, coil_data, resp


def _save_inputs(nii_fmap=None, fname_fmap=None,
                 nii_anat=None, fname_anat=None,
                 nii_mask=None, fname_mask=None,
                 nii_coil=None, fname_coil=None,
                 fm_data=None, fname_fm_json=None,
                 anat_data=None, fname_anat_json=None,
                 coil_data=None, fname_coil_json=None,
                 fname_resp=None, fname_new_resp=None):

    """Save inputs if they are not None, use the respective fnames for the different inputs to save"""
    if nii_fmap is not None:
        # Save the fieldmap
        nib.save(nii_fmap, fname_fmap)

    if fm_data is not None:
        # Save json
        with open(fname_fm_json, 'w', encoding='utf-8') as f:
            json.dump(fm_data, f, indent=4)

    if nii_anat is not None:
        # Save the anat
        nib.save(nii_anat, fname_anat)

    if anat_data is not None:
        # Save json
        with open(fname_anat_json, 'w', encoding='utf-8') as f:
            json.dump(anat_data, f, indent=4)

    if nii_mask is not None:
        # Save the mask
        nib.save(nii_mask, fname_mask)

    if nii_coil is not None:
        # Save the coil
        nib.save(nii_coil, fname_coil)

    if coil_data is not None:
        # Save json
        with open(fname_coil_json, 'w', encoding='utf-8') as f:
            json.dump(coil_data, f, indent=4)

    if fname_resp is not None and fname_new_resp is not None:
        shutil.copy(fname_resp, fname_new_resp)
