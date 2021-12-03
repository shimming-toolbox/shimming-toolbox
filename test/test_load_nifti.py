#!usr/bin/env python3
# -*- coding: utf-8

import json
import nibabel as nib
import numpy as np
import math
import os
import pytest
import shutil

from io import StringIO
from pathlib import Path
from shimmingtoolbox.load_nifti import load_nifti, read_nii
from shimmingtoolbox import __dir_testing__


class TestCore(object):
    _data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                      [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])
    _data_volume = np.array([[[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]], [[7, 7], [8, 8], [9, 9]]],
                             [[[10, 10], [11, 11], [12, 12]], [[13, 13], [14, 14], [15, 15]],
                              [[16, 16], [17, 17], [18, 18]]],
                             [[[19, 19], [20, 20], [21, 21]], [[22, 22], [23, 23], [24, 24]],
                              [[25, 25], [26, 26], [27, 27]]]])
    _data_b1 = np.zeros([64, 64, 5, 16])
    _aff = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    _json_phase = {"Modality": "MR",
                   "ImageComments": "phase",
                   "MagneticFieldStrength": 3,
                   "ImagingFrequency": 123.259,
                   "Manufacturer": "Siemens",
                   "ManufacturersModelName": "Prisma_fit",
                   "InstitutionName": "IUGM",
                   "InstitutionAddress": "",
                   "DeviceSerialNumber": "167006",
                   "StationName": "MRC35049",
                   "BodyPartExamined": "BRAIN",
                   "PatientPosition": "HFS",
                   "ProcedureStepDescription": "dev_acdc",
                   "SoftwareVersions": "syngo_MR_E11",
                   "MRAcquisitionType": "2D",
                   "SeriesDescription": "a_gre_DYNshim",
                   "ProtocolName": "a_gre_DYNshim",
                   "ScanningSequence": "GR",
                   "SequenceVariant": "SP",
                   "SequenceName": "fl2d6",
                   "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND"],
                   "SeriesNumber": 6,
                   "AcquisitionTime": "16:21:2.480000",
                   "AcquisitionNumber": 1,
                   "SliceThickness": 3,
                   "SpacingBetweenSlices": 3,
                   "SAR": 0.00453667,
                   "EchoNumber": 1,
                   "EchoTime": 0.0025,
                   "RepetitionTime": 0.5,
                   "FlipAngle": 25,
                   "PartialFourier": 1,
                   "BaseResolution": 128,
                   "ShimSetting": [620, -7194, -9127, 77, 28, -20, -10, -23],
                   "TxRefAmp": 222.944,
                   "PhaseResolution": 1,
                   "ReceiveCoilName": "HeadNeck_64",
                   "ReceiveCoilActiveElements": "HC7;NC1,2",
                   "PulseSequenceDetails": "%CustomerSeq%_a_gre_DYNshim",
                   "ConsistencyInfo": "N4_VE11C_LATEST_20160120",
                   "PercentPhaseFOV": 59.375,
                   "EchoTrainLength": 6,
                   "PhaseEncodingSteps": 76,
                   "AcquisitionMatrixPE": 76,
                   "ReconMatrixPE": 76,
                   "PixelBandwidth": 600,
                   "PhaseEncodingDirection": "j-",
                   "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                   "InPlanePhaseEncodingDirectionDICOM": "COL",
                   "ConversionSoftware": "dcm2niix",
                   "ConversionSoftwareVersion": "v1.0.20181125  (JP2:OpenJPEG) GCC9.3.0",
                   "Dcm2bidsVersion": "2.1.4"}

    _json_mag = {"Modality": "MR",
                 "ImageComments": "magnitude",
                 "MagneticFieldStrength": 3,
                 "ImagingFrequency": 123.259,
                 "Manufacturer": "Siemens",
                 "ManufacturersModelName": "Prisma_fit",
                 "InstitutionName": "IUGM",
                 "InstitutionAddress": "",
                 "DeviceSerialNumber": "167006",
                 "StationName": "MRC35049",
                 "BodyPartExamined": "BRAIN",
                 "PatientPosition": "HFS",
                 "ProcedureStepDescription": "dev_acdc",
                 "SoftwareVersions": "syngo_MR_E11",
                 "MRAcquisitionType": "2D",
                 "SeriesDescription": "a_gre_DYNshim",
                 "ProtocolName": "a_gre_DYNshim",
                 "ScanningSequence": "GR",
                 "SequenceVariant": "SP",
                 "SequenceName": "fl2d6",
                 "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND"],
                 "SeriesNumber": 6,
                 "AcquisitionTime": "16:21:2.480000",
                 "AcquisitionNumber": 1,
                 "SliceThickness": 3,
                 "SpacingBetweenSlices": 3,
                 "SAR": 0.00453667,
                 "EchoNumber": 1,
                 "EchoTime": 0.0025,
                 "RepetitionTime": 0.5,
                 "FlipAngle": 25,
                 "PartialFourier": 1,
                 "BaseResolution": 128,
                 "ShimSetting": [620, -7194, -9127, 77, 28, -20, -10, -23],
                 "TxRefAmp": 222.944,
                 "PhaseResolution": 1,
                 "ReceiveCoilName": "HeadNeck_64",
                 "ReceiveCoilActiveElements": "HC7;NC1,2",
                 "PulseSequenceDetails": "%CustomerSeq%_a_gre_DYNshim",
                 "ConsistencyInfo": "N4_VE11C_LATEST_20160120",
                 "PercentPhaseFOV": 59.375,
                 "EchoTrainLength": 6,
                 "PhaseEncodingSteps": 76,
                 "AcquisitionMatrixPE": 76,
                 "ReconMatrixPE": 76,
                 "PixelBandwidth": 600,
                 "PhaseEncodingDirection": "j-",
                 "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                 "InPlanePhaseEncodingDirectionDICOM": "COL",
                 "ConversionSoftware": "dcm2niix",
                 "ConversionSoftwareVersion": "v1.0.20181125  (JP2:OpenJPEG) GCC9.3.0",
                 "Dcm2bidsVersion": "2.1.4"}

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
        "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
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
        "ImageOrientationPatientDICOM": [0, 1, 0, 0, 0, -1],
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
        "ImageOrientationPatientDICOM": [1, 0, 0, 0, 0, -1],
        "ImageOrientationText": "Cor",
        "InPlanePhaseEncodingDirectionDICOM": "ROW",
        "ConversionSoftware": "dcm2niix",
        "ConversionSoftwareVersion": "v1.0.20211006"
    }

    def setup_method(self):
        """
        Setup documents for testing of load_nifti
        :return:
        """
        self.full_path = Path(__file__).resolve().parent
        self.test_path = self.full_path

        self.tmp_path = self.test_path / '__temp_nifti__'
        if not self.tmp_path.exists():
            self.tmp_path.mkdir()
        self.toolbox_path = self.test_path.parent

        self.data_path = self.tmp_path / 'test_data'
        self.data_path_2 = self.tmp_path / 'test_data_2'
        self.data_path_volume = self.tmp_path / 'test_data_volume'
        self.data_path_b1 = self.tmp_path / 'test_data_b1'
        self.data_path.mkdir(exist_ok=True)
        self.data_path_2.mkdir(exist_ok=True)
        self.data_path_volume.mkdir(exist_ok=True)
        self.data_path_b1.mkdir(exist_ok=True)

        dummy_data = nib.nifti1.Nifti1Image(dataobj=self._data, affine=self._aff)
        dummy_data_volume = nib.nifti1.Nifti1Image(dataobj=self._data_volume, affine=self._aff)

        nib.save(dummy_data, os.path.join(self.data_path, 'dummy.nii'))
        with open(os.path.join(self.data_path, 'dummy.json'), 'w') as json_file:
            self._json_phase['EchoNumber'] = 1
            json.dump(self._json_phase, json_file)
        nib.save(dummy_data, os.path.join(self.data_path, 'dummy2.nii'))
        with open(os.path.join(self.data_path, 'dummy2.json'), 'w') as json_file:
            self._json_phase['EchoNumber'] = 2
            json.dump(self._json_phase, json_file)

        nib.save(dummy_data, os.path.join(self.data_path_2, 'dummy.nii'))
        with open(os.path.join(self.data_path_2, 'dummy.json'), 'w') as json_file:
            self._json_phase['EchoNumber'] = 1
            json.dump(self._json_phase, json_file)
        nib.save(dummy_data, os.path.join(self.data_path_2, 'dummy2.nii'))
        with open(os.path.join(self.data_path_2, 'dummy2.json'), 'w') as json_file:
            self._json_phase['EchoNumber'] = 2
            json.dump(self._json_phase, json_file)

        nib.save(dummy_data_volume, os.path.join(self.data_path_volume, 'dummy_volume.nii'))
        with open(os.path.join(self.data_path_volume, 'dummy_volume.json'), 'w') as json_file:
            self._json_phase['EchoNumber'] = 1
            json.dump(self._json_phase, json_file)

    def teardown_method(self):
        """
        Tear down for testing of load_nifti
        :return:
        """
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
            pass

    def test_load_nifti_no_folder_fail(self):
        """
        Assert fails without existing path
        :return:
        """
        with pytest.raises(RuntimeError, match="Not an existing NIFTI path"):
            load_nifti("dummy")

    def test_load_nifti_mix_file_types_fail(self):
        """
        Assert fails if folder and files in path
        :return:
        """
        with pytest.raises(RuntimeError, match="Directories and files in input path"):
            load_nifti(str(self.toolbox_path))

    def test_load_nifti_folders(self, monkeypatch):
        """
        Assert that pass if path contains only folders
        :return:
        """
        if self.data_path_2.exists():
            shutil.rmtree(self.data_path_2)
        os.remove(os.path.join(self.data_path, "dummy2.nii"))
        os.remove(os.path.join(self.data_path, "dummy2.json"))
        monkeypatch.setattr('sys.stdin', StringIO('0\n'))
        niftis, info, json_info = load_nifti(str(self.tmp_path))
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"

    def test_load_nifti_files(self):
        """
        Assert that pass if path contains only files
        :return:
        """
        if self.data_path_2.exists():
            shutil.rmtree(self.data_path_2)
        if self.data_path_volume.exists():
            shutil.rmtree(self.data_path_volume)
        os.remove(os.path.join(self.data_path, "dummy2.nii"))
        os.remove(os.path.join(self.data_path, "dummy2.json"))
        niftis, info, json_info = load_nifti(str(self.data_path))
        assert (len(info) == 1), "Wrong number of info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"

    def test_load_nifti_json_missing_fail(self):
        """
        Assert fails if json missing
        :return:
        """
        os.remove(os.path.join(self.data_path, "dummy.json"))
        with pytest.raises(OSError, match="Missing json file"):
            load_nifti(str(self.data_path))

    def test_load_nifti_multiple_echoes(self, monkeypatch):
        """
        Assert passes with correct data for multiple echoes
        :return:
        """
        if self.tmp_path.exists():
            shutil.rmtree(self.data_path_volume)
        monkeypatch.setattr('sys.stdin', StringIO('0\n'))
        niftis, info, json_info = load_nifti(str(self.tmp_path))
        assert (len(info) == 2), "Wrong number od info data 1"
        assert (len(json_info) == 2), "Wrong number of JSON data 1"
        self._json_phase['EchoNumber'] = 1
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON1"
        self._json_phase['EchoNumber'] = 2
        assert (json.dumps(json_info[1], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for second JSON 1"
        assert (niftis.shape == (3, 3, 3, 2, 1)), "Wrong shape for the Nifti output data 1"

        monkeypatch.setattr('sys.stdin', StringIO('1\n'))
        niftis, info, json_info = load_nifti(str(self.tmp_path))
        assert (len(info) == 2), "Wrong number of info data 2"
        assert (len(json_info) == 2), "Wrong number of JSON data 2"
        self._json_phase['EchoNumber'] = 1
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON 2"
        self._json_phase['EchoNumber'] = 2
        assert (json.dumps(json_info[1], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for second JSON 2"
        assert (niftis.shape == (3, 3, 3, 2, 1)), "Wrong shape for the Nifti output data 2"

    def test_load_nifti_quit(self, monkeypatch):
        """
        Assert q quits loading with return 0
        :return:
        """
        monkeypatch.setattr('sys.stdin', StringIO('q\n'))
        ret = load_nifti(str(self.tmp_path))
        assert (ret == 0), "Should have returned 0 for quit input"

    def test_load_nifti_volume(self):
        """
        Assert data containing volume is correctly parsed
        :return:
        """
        if self.data_path_2.exists():
            shutil.rmtree(self.data_path_2)
        if self.data_path.exists():
            shutil.rmtree(self.data_path)
        niftis, info, json_info = load_nifti(str(self.data_path_volume))
        assert (len(info) == 1), "Wrong number of info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 2)), "Wrong shape for the Nifti output data"

    def test_load_nifti_multiple_run(self, monkeypatch):
        """
        Assert data is correctly separated between runs
        :return:
        """
        if self.data_path_2.exists():
            shutil.rmtree(self.data_path_2)
        if self.data_path_volume.exists():
            shutil.rmtree(self.data_path_volume)
        os.remove(os.path.join(self.data_path, "dummy2.nii"))
        os.remove(os.path.join(self.data_path, "dummy2.json"))
        dummy_data = nib.nifti1.Nifti1Image(dataobj=self._data, affine=self._aff)
        nib.save(dummy_data, os.path.join(self.data_path, 'dummy2.nii'))
        with open(os.path.join(self.data_path, 'dummy2.json'), 'w') as json_file:
            self._json_phase['AcquisitionNumber'] = 2
            json.dump(self._json_phase, json_file)

        monkeypatch.setattr('sys.stdin', StringIO('1\n'))
        niftis, info, json_info = load_nifti(str(self.data_path))
        self._json_phase['AcquisitionNumber'] = 1
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"

        monkeypatch.setattr('sys.stdin', StringIO('2\n'))
        niftis, info, json_info = load_nifti(str(self.data_path))
        self._json_phase['AcquisitionNumber'] = 2
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"
        self._json_phase['AcquisitionNumber'] = 1

    def test_load_nifti_modality_check(self, monkeypatch):
        """
        Assert passes with correct data for multiple echoes
        :return:
        """
        if self.data_path_2.exists():
            shutil.rmtree(self.data_path_2)
        if self.data_path_volume.exists():
            shutil.rmtree(self.data_path_volume)
        os.remove(os.path.join(self.data_path, "dummy2.nii"))
        os.remove(os.path.join(self.data_path, "dummy2.json"))
        dummy_data = nib.nifti1.Nifti1Image(dataobj=self._data, affine=self._aff)
        nib.save(dummy_data, os.path.join(self.data_path, 'dummy2.nii'))
        with open(os.path.join(self.data_path, 'dummy2.json'), 'w') as json_file:
            json.dump(self._json_mag, json_file)
        niftis, info, json_info = load_nifti(str(self.data_path))
        assert (len(info) == 1), "Wrong number od info data 1"
        assert (len(json_info) == 1), "Wrong number of JSON data 1"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON1"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data 1"

        niftis, info, json_info = load_nifti(str(self.data_path), "magnitude")
        assert (len(info) == 1), "Wrong number of info data 2"
        assert (len(json_info) == 1), "Wrong number of JSON data 2"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_mag, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON 2"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data 2"

    def test_read_nii_real_data(self):
        fname_phasediff = os.path.join(__dir_testing__, 'ds_b0', 'sub-realtime', 'fmap', 'sub-realtime_phasediff.nii.gz')
        nii, json_info, phasediff = read_nii(fname_phasediff)

        assert nii.shape == (64, 96, 1, 10)
        assert ('P' in json_info['ImageType'])
        assert (phasediff.max() <= math.pi) and (phasediff.min() >= -math.pi)

    def test_read_nii_b1_axial(self):
        fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_axial.nii.gz')
        nii, json_info, b1 = read_nii(fname_b1)

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

    def test_read_nii_b1_coronal(self):
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

    def test_read_nii_b1_sagittal(self):
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

    def test_read_nii_b1_no_orientation_text(self):
        fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_no_orientation_text.nii')
        nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
        self._json_b1_no_orientation = self._json_b1_axial.copy()
        # Remove 'ImageOrientationText' tag from .json file
        del self._json_b1_no_orientation['ImageOrientationText']
        with open(os.path.join(self.data_path_b1, 'dummy_b1_no_orientation_text.json'), 'w') as json_file:
            json.dump(self._json_b1_no_orientation, json_file)
        with pytest.raises(KeyError, match="Missing json tag: 'ImageOrientationText'. Check dcm2niix version."):
            read_nii(fname_b1)

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

    def test_read_nii_b1_no_scaling(self):
        fname_b1 = os.path.join(__dir_testing__, 'ds_tb1', 'sub-tb1tfl', 'fmap', 'sub-tb1tfl_TB1TFL_axial.nii.gz')
        _, _, b1 = read_nii(fname_b1, auto_scale=False)
        assert b1.shape == (64, 44, 5, 16), "Wrong B1 map shape"
        assert [b1[35, 35, 0, 0], b1[35, 35, 2, 13], b1[40, 25, 4, 7]] == [101, 2266, 281]

    def test_read_nii_b1_wrong_shimsetting(self):
        fname_b1 = os.path.join(self.data_path_b1, 'dummy_b1_wrong_shimsetting.nii')
        nib.save(nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff), fname_b1)
        self._json_b1_wrong_shimsetting = self._json_b1_axial.copy()
        # Modify 'ShimSetting' tag in .json tag
        self._json_b1_wrong_shimsetting['ShimSetting'] = str(np.zeros([15]))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_wrong_shimsetting.json'), 'w') as json_file:
            json.dump(self._json_b1_wrong_shimsetting, json_file)
        with pytest.raises(ValueError, match="Wrong array dimension: number of channels not matching"):
            read_nii(fname_b1)

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
