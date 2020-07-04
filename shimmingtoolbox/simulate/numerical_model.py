# coding: utf-8

# Scientific modules imports
import numpy as np

class NumericalModel():

    num_vox = None
    starting_volume = None

    def __init__(self, num_vox=128):

        self.num_vox = num_vox
        self.starting_volume = np.zeros((num_vox, num_vox))

