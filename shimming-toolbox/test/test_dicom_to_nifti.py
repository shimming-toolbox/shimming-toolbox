# coding: utf-8

import copy
import json
import pathlib
import pytest
import tempfile
import shutil

from shimmingtoolbox.dicom_to_nifti import *
from shimmingtoolbox import __dir_testing__

path_dicom_unsorted = os.path.join(__dir_testing__, 'dicom_unsorted')
path_b1_data = os.path.join(__dir_testing__, 'ds_tb1', 'tmp_dcm2bids', 'helper')
fname_b1_nii = os.path.join(path_b1_data, '053_tmp_standard_tra_20210928093219.nii.gz')
fname_b1_json = os.path.join(path_b1_data, '053_tmp_standard_tra_20210928093219.json')
nii_b1 = nib.load(fname_b1_nii)
with open(fname_b1_json) as json_b1_file:
    json_b1 = json.load(json_b1_file)


def test_dicom_to_nifti():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'test'
        dicom_to_nifti(
            path_dicom=path_dicom_unsorted,
            path_nifti=path_nifti,
            subject_id=subject_id
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        for i in range(1, 7):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    assert os.path.exists(os.path.join(path_nifti, f"sub-{subject_id}",
                                                       'target', f"sub-{subject_id}_{modality}{i}.{ext}"))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_realtime_zshim(test_dcm2niix_installation):
    """Test dicom_to_nifti outputs the correct files for realtime_zshimming_data"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'test'
        dicom_to_nifti(
            path_dicom=os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'sourcedata'),
            path_nifti=path_nifti,
            subject_id=subject_id
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.

        sequence_type = 'fmap'
        for i in range(2):
            for modality in ['phase', 'magnitude']:
                for ext in ['nii.gz', 'json']:
                    if modality == 'phase':
                        assert os.path.exists(os.path.join(path_nifti, f"sub-{subject_id}", sequence_type,
                                                           f"sub-{subject_id}_{'phasediff'}.{ext}"))
                    else:
                        assert os.path.exists(os.path.join(path_nifti, f"sub-{subject_id}", sequence_type,
                                                           f"sub-{subject_id}_{modality}{i+1}.{ext}"))

        sequence_type = 'target'
        for i in range(3):
            for ext in ['nii.gz', 'json']:
                assert os.path.exists(
                    os.path.join(path_nifti, f"sub-{subject_id}", sequence_type,
                                 f"sub-{subject_id}_magnitude{i+1}.{ext}"))

        sequence_type = 'func'
        for ext in ['nii.gz', 'json']:
            assert os.path.exists(
                os.path.join(path_nifti, f"sub-{subject_id}", sequence_type, f"sub-{subject_id}_bold.{ext}"))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_remove_tmp(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'test'
        dicom_to_nifti(
            path_dicom=path_dicom_unsorted,
            path_nifti=path_nifti,
            subject_id=subject_id,
            remove_tmp=True
        )
        # Check that all the files (.nii.gz and .json) are created with the expected names. The test data has 6
        # magnitude and phase data.
        assert os.path.exists(path_nifti)
        assert not os.path.exists(os.path.join(path_nifti, 'tmp_dcm2bids'))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_path_dicom_invalid(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_dicom = 'dummy_path'
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'test'
        with pytest.raises(FileNotFoundError, match=r"No dicom path found"):
            dicom_to_nifti(
                path_dicom=path_dicom,
                path_nifti=path_nifti,
                subject_id=subject_id
            )


@pytest.mark.dcm2niix
def test_dicom_to_nifti_path_config_invalid(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'test'
        with pytest.raises(FileNotFoundError, match=r"No dcm2bids config file found"):
            dicom_to_nifti(
                path_dicom=path_dicom_unsorted,
                path_nifti=path_nifti,
                subject_id=subject_id,
                fname_config_dcm2bids=os.path.join(tmp, "invalid_folder")
            )


def test_rename_dual_echo():
    with tempfile.TemporaryDirectory(prefix='st_' + pathlib.Path(__file__).stem) as tmp:
        subject_id = 'fieldmap'
        path_fmap = os.path.join(tmp, f"sub-{subject_id}", 'fmap')
        os.mkdir(os.path.join(tmp, f"sub-{subject_id}"))
        os.mkdir(path_fmap)
        fname_json_e1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.json')
        with open(fname_json_e1) as json_file:
            json_data = json.load(json_file)
        del json_data['EchoTime']
        json_data['EchoTime1'] = 1
        json_data['EchoTime2'] = 2
        json_data['SequenceName'] = 'fl2d2'
        fname_json_e1_new = os.path.join(path_fmap, 'sub-fieldmap_phase1.json')
        with open(fname_json_e1_new, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        fname_json_e2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.json')
        with open(fname_json_e2) as json_file:
            json_data = json.load(json_file)
        del json_data['EchoTime']
        json_data['EchoTime1'] = 1
        json_data['EchoTime2'] = 2
        json_data['SequenceName'] = 'fl2d2'
        fname_json_e2_new = os.path.join(path_fmap, 'sub-fieldmap_phase2.json')
        with open(fname_json_e2_new, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        fname_nii_e1 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase1.nii.gz')
        fname_nii_e1_new = os.path.join(path_fmap, 'sub-fieldmap_phase1.nii.gz')
        shutil.copyfile(fname_nii_e1, fname_nii_e1_new)
        fname_nii_e2 = os.path.join(__dir_testing__, 'ds_b0', 'sub-fieldmap', 'fmap', 'sub-fieldmap_phase2.nii.gz')
        fname_nii_e2_new = os.path.join(path_fmap, 'sub-fieldmap_phase2.nii.gz')
        shutil.copyfile(fname_nii_e2, fname_nii_e2_new)

        rename_phasediff(tmp, subject_id)
        assert os.path.isfile(fname_nii_e2_new)


def test_fix_tfl_b1():
    nii_b1_fixed = fix_tfl_b1(nii_b1, json_b1)
    assert nii_b1_fixed.header['aux_file'] == np.asarray(b'Uncombined B1+ maps', dtype='|S24')


def test_fix_tfl_b1_negative_mag():
    nii_negative_mag = copy.deepcopy(nii_b1)
    b1_negative_value = nii_negative_mag.get_fdata()
    b1_negative_value[35, 35, 0, 0] = -1
    # Set a voxel to an unexpected negative value
    nii_b1_negative_mag = nib.nifti1.Nifti1Image(dataobj=b1_negative_value, affine=np.eye(4))
    with pytest.raises(ValueError, match="Unexpected negative magnitude values"):
        fix_tfl_b1(nii_b1_negative_mag, json_b1)


def test_fix_tfl_b1_no_orientation():
    json_b1_no_orientation = copy.deepcopy(json_b1)
    # Remove 'ImageOrientationPatientDICOM' tag from .json file
    del json_b1_no_orientation['ImageOrientationPatientDICOM']
    with pytest.raises(KeyError, match="Missing JSON tag: 'ImageOrientationPatientDICOM'. Check dcm2niix version."):
        fix_tfl_b1(nii_b1, json_b1_no_orientation)


def test_fix_tfl_b1_no_orientation_text(caplog):
    json_b1_no_orientation_text = copy.deepcopy(json_b1)
    # Remove 'ImageOrientationText' tag in .json file
    del json_b1_no_orientation_text['ImageOrientationText']
    fix_tfl_b1(nii_b1, json_b1_no_orientation_text)
    assert "No 'ImageOrientationText' tag. Slice orientation determined from 'ImageOrientationPatientDICOM'." in \
           caplog.text


def test_fix_tfl_b1_unknown_orientation():
    json_b1_unknown_orientation = copy.deepcopy(json_b1)
    # Modify 'ImageOrientationText' tag in .json file
    json_b1_unknown_orientation['ImageOrientationText'] = 'dummy_string'
    with pytest.raises(ValueError, match="Unknown slice orientation"):
        fix_tfl_b1(nii_b1, json_b1_unknown_orientation)
