#!usr/bin/env python3
# -*- coding: utf-8

import shutil
import os
import numpy as np
import nibabel as nib
import json
import math

from io import StringIO
from pathlib import Path
from shimmingtoolbox.load_nifti import load_nifti
from shimmingtoolbox.load_nifti import read_nii
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
    _data_b1 = np.zeros([64, 64, 16, 16])
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

    _json_b1 = {"Modality": "MR",
                "MagneticFieldStrength": 7,
                "ImagingFrequency": 297.197,
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "Investigational_Device_7T",
                "InstitutionName": "Hospital",
                "InstitutionalDepartmentName": "Department",
                "InstitutionAddress": "Street StreetNo,City,District,CA,ZIP",
                "DeviceSerialNumber": "79017",
                "StationName": "AWP79017",
                "BodyPartExamined": "BRAIN",
                "PatientPosition": "HFS",
                "ProcedureStepDescription": "Development^Dr. Cohen-Adad",
                "SoftwareVersions": "syngo MR E12",
                "MRAcquisitionType": "2D",
                "SeriesDescription": "tfl_rfmap_B1shim_flip20_5mm",
                "ProtocolName": "tfl_rfmap_B1shim_flip20_5mm",
                "ScanningSequence": "GR",
                "SequenceVariant": "SK\\SP",
                "SequenceName": "tfl2d1_16",
                "SeriesNumber": 67,
                "AcquisitionTime": "14:21:12.127500",
                "AcquisitionNumber": 1,
                "ImageComments": "flip angle map, TraRefAmpl: 225.0 V",
                "SliceThickness": 5,
                "SpacingBetweenSlices": 6,
                "SAR": 0.0250297,
                "EchoTime": 0.00148,
                "RepetitionTime": 4,
                "FlipAngle": 20,
                "PartialFourier": 1,
                "Interpolation2D": 1,
                "BaseResolution": 32,
                "ShimSetting": [133, 30, 22, -80, 27, -130, -112, -234],
                "TxRefAmp": 225,
                "PhaseResolution": 1,
                "ReceiveCoilName": "NP11_ACDC_SPINE",
                "ReceiveCoilActiveElements": "1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H;1H",
                "PulseSequenceDetails": "%SiemensSeq%\\tfl_rfmap",
                "ConsistencyInfo": "N4_VE12U_LATEST_20181126",
                "MultibandAccelerationFactor": 8,
                "PercentPhaseFOV": 100,
                "PercentSampling": 100,
                "PhaseEncodingSteps": 32,
                "AcquisitionMatrixPE": 32,
                "ReconMatrixPE": 64,
                "PixelBandwidth": 450,
                "DwellTime": 3.48e-05,
                "PhaseEncodingDirection": "j-",
                "SliceTiming": [0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0, 0, 0, 0, 0,
                                0, 0, 0],
                "ImageOrientationPatientDICOM": [1, 0, 0, 0, 1, 0],
                "InPlanePhaseEncodingDirectionDICOM": "COL",
                "ConversionSoftware": "dcm2niix",
                "ConversionSoftwareVersion": "v1.0.20201102",
                "Dcm2bidsVersion": "2.1.4"}

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
        try:
            load_nifti("dummy")
        except RuntimeError:
            return 0

        assert False, "Did not fail if no valid path given"

    def test_load_nifti_mix_file_types_fail(self):
        """
        Assert fails if folder and files in path
        :return:
        """
        try:
            load_nifti(self.toolbox_path)
        except:
            return 0

        assert False, "Did not fail with folder and files in the same path"

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
        niftis, info, json_info = load_nifti(self.tmp_path)
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)),\
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
        niftis, info, json_info = load_nifti(self.data_path)
        assert (len(info) == 1), "Wrong number of info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)),\
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"

    def test_load_nifti_json_missing_fail(self):
        """
        Assert fails if json missing
        :return:
        """
        os.remove(os.path.join(self.data_path, "dummy.json"))
        try:
            load_nifti(self.data_path)
        except ValueError:
            return 0

        assert False, "Did not fail with missing JSON file"

    def test_load_nifti_multiple_echoes(self, monkeypatch):
        """
        Assert passes with correct data for multiple echoes
        :return:
        """
        if self.tmp_path.exists():
            shutil.rmtree(self.data_path_volume)
        monkeypatch.setattr('sys.stdin', StringIO('0\n'))
        niftis, info, json_info = load_nifti(self.tmp_path)
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
        niftis, info, json_info = load_nifti(self.tmp_path)
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
        ret = load_nifti(self.tmp_path)
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
        niftis, info, json_info = load_nifti(self.data_path_volume)
        assert (len(info) == 1), "Wrong number of info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)),\
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
        niftis, info, json_info = load_nifti(self.data_path)
        self._json_phase['AcquisitionNumber'] = 1
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)),\
            "JSON file is not correctly loaded"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data"

        monkeypatch.setattr('sys.stdin', StringIO('2\n'))
        niftis, info, json_info = load_nifti(self.data_path)
        self._json_phase['AcquisitionNumber'] = 2
        assert (len(info) == 1), "Wrong number od info data"
        assert (len(json_info) == 1), "Wrong number of JSON data"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)),\
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
        niftis, info, json_info = load_nifti(self.data_path)
        assert (len(info) == 1), "Wrong number od info data 1"
        assert (len(json_info) == 1), "Wrong number of JSON data 1"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_phase, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON1"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data 1"

        niftis, info, json_info = load_nifti(self.data_path, "magnitude")
        assert (len(info) == 1), "Wrong number of info data 2"
        assert (len(json_info) == 1), "Wrong number of JSON data 2"
        assert (json.dumps(json_info[0], sort_keys=True) == json.dumps(self._json_mag, sort_keys=True)), \
            "JSON file is not correctly loaded for first JSON 2"
        assert (niftis.shape == (3, 3, 3, 1, 1)), "Wrong shape for the Nifti output data 2"

    def test_read_nii_real_data(self):
        fname_phasediff = os.path.join(__dir_testing__, 'realtime_zshimming_data', 'nifti', 'sub-example', 'fmap',
                                       'sub-example_phasediff.nii.gz')
        nii, json_info, phasediff = read_nii(fname_phasediff)

        assert nii.shape == (64, 96, 1, 10)
        assert ('P' in json_info['ImageType'])
        assert (phasediff.max() <= 2 * math.pi) and (phasediff.min() >= 0)

    def test_read_nii_b1(self):
        fname_b1 = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'sub-01_run-10_TB1map.nii.gz')
        nii, json_info, b1 = read_nii(fname_b1)

        assert b1.shape == (64, 64, 16, 8), "Wrong rf-map shape"
        assert np.abs(b1).max() <= 180 and np.abs(b1).min() >= 0, "Magnitude values out of range"
        assert np.angle(b1).max(initial=0) <= np.pi and np.angle(b1).min(initial=0) >= -np.pi,\
            "Phase values out of range"

        # Check masking consistency for all coils at each slice
        for i in range(b1.shape[2]):
            for j in range(b1.shape[3] - 1):
                assert ((b1[:, :, i, j] != 0) == (b1[:, :, i, j + 1] != 0)).any()

        test_values = [-4.274539911369111 + 4.599952786001116j,
                       -5.8027003257021725 + 2.2042390773527423j,
                       -2.1929304691258276 + 1.5241263801971388j]

        assert np.isclose([b1[35, 35, 0, 0], b1[35, 35, 6, 7], b1[40, 25, 15, 7]], test_values).all()

        assert (json.dumps(json_info, sort_keys=True) == json.dumps(self._json_b1, sort_keys=True)),\
            "JSON file is not correctly loaded for first RF JSON"

    def test_read_nii_b1_without_tags(self):
        dummy_data_b1 = nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff)
        nib.save(dummy_data_b1, os.path.join(self.data_path_b1, 'dummy_b1_no_shimsetting'))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_no_shimsetting.json'), 'w') as json_file:
            self._json_b1_no_shimsetting = self._json_b1.copy()
            del self._json_b1_no_shimsetting['ShimSetting']
            json.dump(self._json_b1_no_shimsetting, json_file)

        fname_b1 = os.path.join(self.data_path_b1, "dummy_b1_no_shimsetting.nii")
        try:
            read_nii(fname_b1)
        except ValueError:
            return 0

        nib.save(dummy_data_b1, os.path.join(self.data_path_b1, 'dummy_b1_no_slicetiming'))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_no_slicetiming.json'), 'w') as json_file:
            self._json_b1_no_slicetiming = self._json_b1.copy()
            del self._json_b1_no_slicetiming['SliceTiming']
            json.dump(self._json_b1_no_slicetiming, json_file)

        fname_b1 = os.path.join(self.data_path_b1, "dummy_b1_no_slicetiming.nii")
        try:
            read_nii(fname_b1)
        except ValueError:
            return 0

    def test_read_nii_b1_no_scaling(self):
        fname_b1 = os.path.join(__dir_testing__, 'b1_maps', 'nifti', 'sub-01_run-10_TB1map.nii.gz')
        _, _, b1 = read_nii(fname_b1, auto_scale=False)
        assert b1.shape == (64, 64, 16, 16), "Wrong rf-map shape"
        test_values = [87.0, 1890.0, 37.0]
        assert [b1[35, 35, 0, 0], b1[35, 35, 6, 13], b1[40, 25, 15, 7]] == test_values

    def test_read_nii_b1_wrong_dims(self):
        dummy_data_b1 = nib.nifti1.Nifti1Image(dataobj=self._data_b1, affine=self._aff)
        nib.save(dummy_data_b1, os.path.join(self.data_path_b1, 'dummy_b1_wrong_shimsetting'))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_wrong_shimsetting.json'), 'w') as json_file:
            self._json_b1_wrong_shimsetting = self._json_b1.copy()
            self._json_b1_wrong_shimsetting['ShimSetting'] = str(np.zeros([15]))
            json.dump(self._json_b1_wrong_shimsetting, json_file)

        fname_b1 = os.path.join(self.data_path_b1, "dummy_b1_wrong_shimsetting.nii")
        try:
            read_nii(fname_b1)
        except ValueError:
            return 0

        nib.save(dummy_data_b1, os.path.join(self.data_path_b1, 'dummy_b1_wrong_slicetiming'))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_wrong_slicetiming.json'), 'w') as json_file:
            self._json_b1_wrong_slicetiming = self._json_b1.copy()
            self._json_b1_wrong_shimsetting['SliceTiming'] = str(np.zeros([15]))
            json.dump(self._json_b1_wrong_slicetiming, json_file)

        fname_b1 = os.path.join(self.data_path_b1, "dummy_b1_wrong_slicetiming.nii")
        try:
            read_nii(fname_b1)
        except ValueError:
            return 0

    def test_read_nii_b1_negative_mag(self):
        data_negative_mag = self._data_b1.copy()
        data_negative_mag[35, 35, 0, 0] = -1
        dummy_data_b1 = nib.nifti1.Nifti1Image(dataobj=data_negative_mag, affine=self._aff)
        nib.save(dummy_data_b1, os.path.join(self.data_path_b1, 'dummy_b1_negative_mag'))
        with open(os.path.join(self.data_path_b1, 'dummy_b1_negative_mag.json'), 'w') as json_file:
            json.dump(self._json_b1, json_file)

        fname_b1 = os.path.join(self.data_path_b1, "dummy_b1_negative_mag.nii")
        try:
            read_nii(fname_b1)
        except ValueError:
            return 0
