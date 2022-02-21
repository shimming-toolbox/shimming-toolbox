# coding: utf-8

import os
import pathlib
import tempfile

from shimmingtoolbox.dicom_to_nifti import dicom_to_nifti
from shimmingtoolbox import __dir_testing__
import pytest

path_dicom_unsorted = os.path.join(__dir_testing__, 'dicom_unsorted')

_data_b1 = np.zeros([64, 64, 5, 16])
_json_b1_axial = {
    "Modality": "MR",
    "MagneticFieldStrength": 7,
    "ImagingFrequency": 297.2,
    "Manufacturer": "Siemens",
    "ManufacturersModelName": "Investigational_Device_7T",
    "InstitutionName": "Hospital",
    "InstitutionalDepartmentName": "Department",
    "InstitutionAddress": "Street StreetNo,City,District,CA,ZIP",
    "DeviceSerialNumber": "79017",
    "StationName": "AWP79017",
    "BodyPartExamined": "CSPINE",
    "PatientPosition": "HFS",
    "ProcedureStepDescription": "Development^Dr. Cohen-Adad",
    "SoftwareVersions": "syngo MR E12",
    "MRAcquisitionType": "2D",
    "SeriesDescription": "standard_tra",
    "ProtocolName": "standard_tra",
    "ScanningSequence": "GR",
    "SequenceVariant": "SK\\SP",
    "SequenceName": "*tfl2d1_16",
    "SeriesNumber": 53,
    "AcquisitionTime": "12:16:59.472500",
    "AcquisitionNumber": 1,
    "ImageComments": "flip angle map, TraRefAmpl: 400.0 V",
    "SliceThickness": 7,
    "SpacingBetweenSlices": 14,
    "SAR": 0.00443249,
    "EchoTime": 0.00153,
    "RepetitionTime": 3.76,
    "SpoilingState": True,
    "FlipAngle": 5,
    "PartialFourier": 1,
    "BaseResolution": 64,
    "ShimSetting": [133, -20, 10, -75, 9, -147, -112, -224],
    "TxRefAmp": 400,
    "PhaseResolution": 1,
    "ReceiveCoilName": "NP11_ACDC_SPINE",
    "ReceiveCoilActiveElements": "1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H",
    "PulseSequenceDetails": "%SiemensSeq%\\tfl_rfmap",
    "RefLinesPE": 16,
    "CoilCombinationMethod": "Sum of Squares",
    "ConsistencyInfo": "N4_VE12U_LATEST_20181126",
    "MatrixCoilMode": "GRAPPA",
    "MultibandAccelerationFactor": 5,
    "PercentPhaseFOV": 68.75,
    "PercentSampling": 100,
    "PhaseEncodingSteps": 44,
    "AcquisitionMatrixPE": 44,
    "ReconMatrixPE": 44,
    "ParallelReductionFactorInPlane": 2,
    "PixelBandwidth": 440,
    "DwellTime": 1.78e-05,
    "PhaseEncodingDirection": "j-",
    "ImageOrientationPatientDICOM": [-1, 0, 0, 0, 1, 0],
    "ImageOrientationText": "Tra",
    "InPlanePhaseEncodingDirectionDICOM": "COL",
    "ConversionSoftware": "dcm2niix",
    "ConversionSoftwareVersion": "v1.0.20211006"
}

_json_b1_sagittal = {
    "Modality": "MR",
    "MagneticFieldStrength": 7,
    "ImagingFrequency": 297.2,
    "Manufacturer": "Siemens",
    "ManufacturersModelName": "Investigational_Device_7T",
    "InstitutionName": "Hospital",
    "InstitutionalDepartmentName": "Department",
    "InstitutionAddress": "Street StreetNo,City,District,CA,ZIP",
    "DeviceSerialNumber": "79017",
    "StationName": "AWP79017",
    "BodyPartExamined": "CSPINE",
    "PatientPosition": "HFS",
    "ProcedureStepDescription": "Development^Dr. Cohen-Adad",
    "SoftwareVersions": "syngo MR E12",
    "MRAcquisitionType": "2D",
    "SeriesDescription": "standard_sag",
    "ProtocolName": "standard_sag",
    "ScanningSequence": "GR",
    "SequenceVariant": "SK\\SP",
    "SequenceName": "*tfl2d1_16",
    "SeriesNumber": 55,
    "AcquisitionTime": "12:21:7.597500",
    "AcquisitionNumber": 1,
    "ImageComments": "flip angle map, TraRefAmpl: 400.0 V",
    "SliceThickness": 5,
    "SpacingBetweenSlices": 10,
    "SAR": 0.00464767,
    "EchoTime": 0.00163,
    "RepetitionTime": 3.76,
    "SpoilingState": True,
    "FlipAngle": 5,
    "PartialFourier": 1,
    "BaseResolution": 64,
    "ShimSetting": [133, -20, 10, -75, 9, -147, -112, -224],
    "TxRefAmp": 400,
    "PhaseResolution": 1,
    "ReceiveCoilName": "NP11_ACDC_SPINE",
    "ReceiveCoilActiveElements": "1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H",
    "PulseSequenceDetails": "%SiemensSeq%\\tfl_rfmap",
    "RefLinesPE": 16,
    "CoilCombinationMethod": "Sum of Squares",
    "ConsistencyInfo": "N4_VE12U_LATEST_20181126",
    "MatrixCoilMode": "GRAPPA",
    "MultibandAccelerationFactor": 5,
    "PercentPhaseFOV": 81.25,
    "PercentSampling": 100,
    "PhaseEncodingSteps": 52,
    "AcquisitionMatrixPE": 52,
    "ReconMatrixPE": 52,
    "ParallelReductionFactorInPlane": 2,
    "PixelBandwidth": 440,
    "DwellTime": 1.78e-05,
    "PhaseEncodingDirection": "i",
    "ImageOrientationPatientDICOM": [0, -1, 0, 0, 0, 1],
    "ImageOrientationText": "Sag",
    "InPlanePhaseEncodingDirectionDICOM": "ROW",
    "ConversionSoftware": "dcm2niix",
    "ConversionSoftwareVersion": "v1.0.20211006"
}

_json_b1_coronal = {
    "Modality": "MR",
    "MagneticFieldStrength": 7,
    "ImagingFrequency": 297.2,
    "Manufacturer": "Siemens",
    "ManufacturersModelName": "Investigational_Device_7T",
    "InstitutionName": "Hospital",
    "InstitutionalDepartmentName": "Department",
    "InstitutionAddress": "Street StreetNo,City,District,CA,ZIP",
    "DeviceSerialNumber": "79017",
    "StationName": "AWP79017",
    "BodyPartExamined": "CSPINE",
    "PatientPosition": "HFS",
    "ProcedureStepDescription": "Development^Dr. Cohen-Adad",
    "SoftwareVersions": "syngo MR E12",
    "MRAcquisitionType": "2D",
    "SeriesDescription": "standard_cor",
    "ProtocolName": "standard_cor",
    "ScanningSequence": "GR",
    "SequenceVariant": "SK\\SP",
    "SequenceName": "*tfl2d1_16",
    "SeriesNumber": 54,
    "AcquisitionTime": "12:19:4.552500",
    "AcquisitionNumber": 1,
    "ImageComments": "flip angle map, TraRefAmpl: 400.0 V",
    "SliceThickness": 5,
    "SpacingBetweenSlices": 10,
    "SAR": 0.00701462,
    "EchoTime": 0.00163,
    "RepetitionTime": 3.76,
    "SpoilingState": True,
    "FlipAngle": 5,
    "PartialFourier": 1,
    "BaseResolution": 64,
    "ShimSetting": [133, -20, 10, -75, 9, -147, -112, -224],
    "TxRefAmp": 400,
    "PhaseResolution": 1,
    "ReceiveCoilName": "NP11_ACDC_SPINE",
    "ReceiveCoilActiveElements": "1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H",
    "PulseSequenceDetails": "%SiemensSeq%\\tfl_rfmap",
    "RefLinesPE": 16,
    "CoilCombinationMethod": "Sum of Squares",
    "ConsistencyInfo": "N4_VE12U_LATEST_20181126",
    "MatrixCoilMode": "GRAPPA",
    "MultibandAccelerationFactor": 5,
    "PercentPhaseFOV": 218.75,
    "PercentSampling": 100,
    "PhaseEncodingSteps": 140,
    "AcquisitionMatrixPE": 140,
    "ReconMatrixPE": 140,
    "ParallelReductionFactorInPlane": 2,
    "PixelBandwidth": 440,
    "DwellTime": 1.78e-05,
    "PhaseEncodingDirection": "i",
    "ImageOrientationPatientDICOM": [-1, 0, 0, 0, 0, 1],
    "ImageOrientationText": "Cor",
    "InPlanePhaseEncodingDirectionDICOM": "ROW",
    "ConversionSoftware": "dcm2niix",
    "ConversionSoftwareVersion": "v1.0.20211006"
}


def test_dicom_to_nifti():
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
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
                    assert os.path.exists(os.path.join(path_nifti, subject_id, 'fmap', subject_id + '_{}{}.{}'.format(
                        modality, i, ext)))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_realtime_zshim(test_dcm2niix_installation):
    """Test dicom_to_nifti outputs the correct files for realtime_zshimming_data"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
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
                        assert os.path.exists(os.path.join(path_nifti, subject_id, sequence_type,
                                                           subject_id + f"_{'phasediff'}.{ext}"))
                    else:
                        assert os.path.exists(os.path.join(path_nifti, subject_id, sequence_type,
                                                           subject_id + f"_{modality}{i+1}.{ext}"))

        sequence_type = 'anat'
        for i in range(3):
            for ext in ['nii.gz', 'json']:
                assert os.path.exists(
                    os.path.join(path_nifti, subject_id, sequence_type, subject_id + f"_unshimmed_e{i+1}.{ext}"))

        sequence_type = 'func'
        for ext in ['nii.gz', 'json']:
            assert os.path.exists(
                os.path.join(path_nifti, subject_id, sequence_type, subject_id + f"_bold.{ext}"))


@pytest.mark.dcm2niix
def test_dicom_to_nifti_remove_tmp(test_dcm2niix_installation):
    """Test the remove_tmp folder"""
    with tempfile.TemporaryDirectory(prefix='st_'+pathlib.Path(__file__).stem) as tmp:
        path_nifti = os.path.join(tmp, 'nifti')
        subject_id = 'sub-test'
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
        subject_id = 'sub-test'
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
        subject_id = 'sub-test'
        with pytest.raises(FileNotFoundError, match=r"No dcm2bids config file found"):
            dicom_to_nifti(
                path_dicom=path_dicom_unsorted,
                path_nifti=path_nifti,
                subject_id=subject_id,
                path_config_dcm2bids=os.path.join(tmp, "invalid_folder")
            )


@pytest.mark.dcm2niix
def test_dicom_to_nifti_tfl_b1_axial(self):
    fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_axial.nii.gz')
    nib.load(fname_b1)
    

    assert b1.shape == (64, 44, 5, 8), "Wrong rf-map shape"
    assert np.abs(b1).max() <= 180 and np.abs(b1).min() >= 0, "Magnitude values out of range"
    assert np.angle(b1).max(initial=0) <= np.pi and np.angle(b1).min(initial=0) >= -np.pi, \
        "Phase values out of range"

    test_values = [0.0787205885749701 + 4.099821199410974j,
                   8.426583467014298 + 4.728778099763556j,
                   10.485988324410084 + 4.494300336459402j]

    assert np.isclose([b1[35, 35, 0, 0], b1[35, 35, 1, 4], b1[40, 25, 4, 7]], test_values).all()
    assert (json.dumps(json_info, sort_keys=True) == json.dumps(self._json_b1_axial, sort_keys=True)), \
        "JSON file is not correctly loaded for first RF JSON"


@pytest.mark.dcm2niix
def test_dicom_to_nifti_tfl_b1_coronal(self):
    fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_coronal.nii.gz')
    nii, json_info, b1 = read_nii(fname_b1)

    assert b1.shape == (140, 64, 5, 8), "Wrong rf-map shape"
    assert np.abs(b1).max() <= 180 and np.abs(b1).min() >= 0, "Magnitude values out of range"
    assert np.angle(b1).max(initial=0) <= np.pi and np.angle(b1).min(initial=0) >= -np.pi, \
        "Phase values out of range"

    test_values = [-18.95330780647338 - 15.623901256474788j, 0j, -4.017854608159664 + 14.390338163103701j]

    assert np.isclose([b1[35, 35, 0, 0], b1[35, 35, 1, 4], b1[40, 25, 4, 7]], test_values).all()
    assert (json.dumps(json_info, sort_keys=True) == json.dumps(self._json_b1_coronal, sort_keys=True)), \
        "JSON file is not correctly loaded for first RF JSON"


@pytest.mark.dcm2niix
def test_dicom_to_nifti_tfl_sagittal(self):
    fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_sagittal.nii.gz')
    nii, json_info, b1 = read_nii(fname_b1)

    assert b1.shape == (52, 64, 5, 8), "Wrong rf-map shape"
    assert np.abs(b1).max() <= 180 and np.abs(b1).min() >= 0, "Magnitude values out of range"
    assert np.angle(b1).max(initial=0) <= np.pi and np.angle(b1).min(initial=0) >= -np.pi, \
        "Phase values out of range"

    test_values = [-2.3972261793386425 - 2.757693261674301j,
                   12.039283903012375 + 4.549266291277882j,
                   7.2905022476747625 + 8.240413764304524j]

    assert np.isclose([b1[35, 35, 0, 0], b1[35, 35, 1, 4], b1[40, 25, 4, 7]], test_values).all()
    assert (json.dumps(json_info, sort_keys=True) == json.dumps(self._json_b1_sagittal, sort_keys=True)), \
        "JSON file is not correctly loaded for first RF JSON"


@pytest.mark.dcm2niix
def test_read_nii_b1_no_orientation(self):
    fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_no_orientation.nii')
    nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
    self._json_b1_no_orientation = self._json_b1_axial.copy()
    # Remove 'ImageOrientationPatientDICOM' tag from .json file
    del self._json_b1_no_orientation['ImageOrientationPatientDICOM']
    with open(os.path.join(self.data_path_b1, 'dummy_b1_no_orientation.json'), 'w') as json_file:
        json.dump(self._json_b1_no_orientation, json_file)
    with pytest.raises(KeyError, match="Missing json tag: 'ImageOrientationPatientDICOM'. Check dcm2niix version."):
        read_nii(fname_b1)


@pytest.mark.dcm2niix
def test_read_nii_b1_no_orientation_text(self, caplog):
    fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_no_orientation_text.nii')
    nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
    self._json_b1_no_orientation = self._json_b1_axial.copy()
    # Remove 'ImageOrientationText' tag from .json file
    del self._json_b1_no_orientation['ImageOrientationText']
    with open(os.path.join(self.data_path_b1, 'dummy_b1_no_orientation_text.json'), 'w') as json_file:
        json.dump(self._json_b1_no_orientation, json_file)

    read_nii(fname_b1)
    assert "No 'ImageOrientationText' tag. Slice orientation determined from 'ImageOrientationPatientDICOM'." in \
           caplog.text


@pytest.mark.dcm2niix
def test_read_nii_b1_unknown_orientation(self):
    fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_unknown_orientation.nii')
    nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
    self._json_b1_unknown_orientation = self._json_b1_axial.copy()
    # Modify 'ImageOrientationText' tag in .json file
    self._json_b1_unknown_orientation['ImageOrientationText'] = 'dummy_string'
    with open(os.path.join(self.data_path_b1, 'dummy_b1_unknown_orientation.json'), 'w') as json_file:
        json.dump(self._json_b1_unknown_orientation, json_file)
    with pytest.raises(ValueError, match="Unknown slice orientation"):
        read_nii(fname_b1)


@pytest.mark.dcm2niix
def test_read_nii_b1_no_scaling(self):
    fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_axial.nii.gz')
    _, _, b1 = read_nii(fname_b1, auto_scale=False)
    assert b1.shape == (64, 44, 5, 16), "Wrong B1 map shape"
    assert [b1[35, 35, 0, 0], b1[35, 35, 2, 13], b1[40, 25, 4, 7]] == [101, 2266, 281]


@pytest.mark.dcm2niix
def test_read_nii_b1_no_shimsetting(self):
    fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_no_shimsetting.nii')
    nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
    self._json_b1_no_shimsetting = self._json_b1_axial.copy()
    # Remove 'ShimSetting' tag from .json file
    del self._json_b1_no_shimsetting['ShimSetting']
    with open(os.path.join(self.data_path_b1, 'dummy_b1_no_shimsetting.json'), 'w') as json_file:
        json.dump(self._json_b1_no_shimsetting, json_file)
    with pytest.raises(KeyError, match="Missing json tag: 'ShimSetting'"):
        read_nii(fname_b1)


@pytest.mark.dcm2niix
def test_read_nii_b1_negative_mag(self):
    fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_negative_mag.nii')
    data_negative_mag = self._data_b1.copy()
    # Set a voxel to an unexpected negative value
    data_negative_mag[35, 35, 0, 0] = -1
    nib.save(nib.nifti1.Nifti1Image(dataobj=data_negative_mag, affine=self._aff), fname_b1)
    with open(os.path.join(self.data_path_b1, 'dummy_b1_negative_mag.json'), 'w') as json_file:
        json.dump(self._json_b1_axial, json_file)
    with pytest.raises(ValueError, match="Unexpected negative magnitude values"):
        read_nii(fname_b1)