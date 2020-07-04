# coding: utf-8

# Scientific modules imports
import os
from pathlib import Path
import json
import numpy as np
from scipy.io import savemat
from copy import deepcopy
from phantominator import shepp_logan
import nibabel as nib

np.seterr(divide='ignore', invalid='ignore')

class NumericalModel():

    gamma = 267.52218744 * 10**6 # rad*Hz/Tesla
    field_strength = 3.0; # Tesla
    handedness = 'left' # Siemens & Canon = 'left', GE & Philips = 'right' 

    num_vox = None
    starting_volume = None
    volume = {
        'T2_star': None,
        'proton_density': None
        }

    # Default brain T2* values in seconds @ 3T
    T2_star = {'WM': 0.053, 'GM': 0.066, 'CSF': 0.10}
        
    # Default brain proton density in percentage
    proton_density = {'WM': 70, 'GM': 82, 'CSF': 100}

    def __init__(self, model=None, num_vox=128):

        self.num_vox = num_vox

        if model is None:
            self.starting_volume = np.zeros((num_vox, num_vox))
        elif model=='shepp-logan':
            self.__shepp_logan_brain__(num_vox)
        
        self.deltaB0 = self.starting_volume * 0

    def __shepp_logan_brain__(self, numVox):
        self.starting_volume = shepp_logan(numVox)

        self.volume['proton_density'] = self.__customize_shepp_logan__(
            self.starting_volume,
            self.proton_density['WM'],
            self.proton_density['GM'],
            self.proton_density['CSF']
            )
        self.volume['T2_star'] = self.__customize_shepp_logan__(
            self.starting_volume,
            self.T2_star['WM'],
            self.T2_star['GM'],
            self.T2_star['CSF']
            )

    def generate_deltaB0(self, fieldType, params):
            
        if fieldType == 'linear':
            m = params[0]
            b = params[1]
                    
            dims = self.starting_volume.shape

            [X, Y] = np.meshgrid(np.linspace(-dims[0], dims[0], dims[0]), np.linspace(-dims[1], dims[1], dims[1]))

            self.deltaB0 = m*X+b

            self.deltaB0 = self.deltaB0 / (self.gamma / (2*np.pi))
        else:
            Exception('Undefined deltaB0 field type')

    def __customize_shepp_logan__(
        self,
        volume,
        class1,
        class2,
        class3
        ):
        
        customVolume = deepcopy(volume)
            
        customVolume[abs(volume-0.2)<0.001] = class1
        customVolume[abs(volume-0.3)<0.001] = class2
        customVolume[abs(volume-1)<0.001] = class3

        customVolume[np.logical_and((abs(volume)<0.0001), volume!=0)] = class1/2
        customVolume[abs(volume-0.1)<0.001] = (class2 + class1)/2
        customVolume[abs(volume-0.4)<0.001] = class2*1.5

        return customVolume

    def simulate_measurement(self, FA, TE, SNR=None):

        self.FA = FA
        self.TE = TE

        numTE = len(TE)
        volDims = self.starting_volume.shape

        if len(volDims) == 2:
            self.measurement = np.zeros((volDims[0], volDims[1], 1, numTE), dtype = 'complex_')
        elif len(volDims) == 3:
            self.measurement = np.zeros((volDims[0], volDims[1], volDims[2], numTE), dtype = 'complex_')

        for ii in range(0,numTE):
            if len(volDims) == 2:
                self.measurement[:,:,0,ii] = self.generate_signal(
                    np.squeeze(self.volume['proton_density']),
                    np.squeeze(self.volume['T2_star']),
                    FA,
                    TE[ii],
                    np.squeeze(self.deltaB0),
                    self.gamma,
                    self.handedness
                    )
            else:
                self.measurement[:,:,:,ii] = self.generate_signal(
                    np.squeeze(self.volume['proton_density']),
                    np.squeeze(self.volume['T2_star']),
                    FA,
                    TE[ii],
                    np.squeeze(self.deltaB0),
                    self.gamma,
                    self.handedness
                    )

        if SNR is not None:
            self.measurement = NumericalModel.add_noise(self.measurement, SNR)

    @staticmethod  
    def generate_signal(
        proton_density,
        T2_star,
        FA,
        TE,
        deltaB0,
        gamma,
        handedness
        ):

        if handedness == 'left':
            sign = -1
        elif 'right':
            sign = 1

        deltaB0 = np.array(deltaB0)
        volDims = deltaB0.shape

        if len(volDims) == 2:
            signal = np.zeros((volDims[0], volDims[1], 1), dtype = 'complex_')
        elif len(volDims) == 3:
            signal = np.zeros((volDims[0], volDims[1], volDims[2]), dtype = 'complex_')

        signal = proton_density*np.sin(np.deg2rad(FA))*np.exp(-TE/T2_star-sign*1j*gamma*deltaB0*TE)

        return signal

    def get_magnitude(self):
        # Get magnitude data
        return np.abs(self.measurement)

    def get_phase(self):
        # Get phase data
        return np.angle(self.measurement)

    def get_real(self):
        # Get real data
        return np.real(self.measurement)

    def get_imaginary(self):
        # Get imaginary data
        return np.imag(self.measurement)
    
    @staticmethod  
    def add_noise(volume, SNR):
        noiseSTD = np.max(volume[:])/SNR

        volDims = volume.shape
        noisyReal =      np.real(volume) + np.random.randn(volDims[0], volDims[1], volDims[2], volDims[3]) * noiseSTD
        noisyImaginary = np.imag(volume) + np.random.randn(volDims[0], volDims[1], volDims[2], volDims[3]) * noiseSTD

        noisyVolume = noisyReal + 1j*noisyImaginary
        return noisyVolume

    def save(self, dataType, fileName, saveFormat=None):

        if saveFormat is None:
            saveFormat = 'nifti'

        if fileName[-4:] == '.nii':
            if saveFormat != 'nifti':
                print('File extension and saveFormat do not match - saving to NIfTI format')
                saveFormat = 'nifti'
            fileName = fileName[0:-4]
        elif fileName[-4:] == '.mat':
            if saveFormat != 'mat':
                print('File extension and saveFormat do not match - saving to MAT format')
                saveFormat = 'mat'
            fileName = fileName[0:-4]

        if dataType == 'Magnitude':
            vol = self.get_magnitude()
        elif dataType == 'Phase':
            vol = self.get_phase()
        elif dataType == 'Real':
            vol = self.get_real()
        elif dataType == 'Imaginary':
            vol = self.get_imaginary()
        else:
            Exception('Unknown datatype')

        if saveFormat == 'nifti':
            empty_header = nib.Nifti1Header()

            img = nib.Nifti1Image(np.rot90(vol), affine=np.eye(4), header=empty_header)
            nib.save(img, Path(fileName + '.nii'))

            self.writeJson(fileName)
        elif saveFormat == 'mat':
            savemat(Path(fileName + '.mat'), {'vol': vol})
            self.writeJson(fileName)

    def writeJson(self, fileName):
        pulseSeqProperties = {
            "EchoTime": self.TE,
            "FlipAngle": self.FA
            }

        with open(Path(fileName + '.json'), 'w') as outfile:
            json.dump(pulseSeqProperties, outfile)
