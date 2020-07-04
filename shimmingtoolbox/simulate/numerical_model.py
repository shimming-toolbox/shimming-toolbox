# coding: utf-8

# Scientific modules imports
import numpy as np
from phantominator import shepp_logan

class NumericalModel():

    num_vox = None
    starting_volume = None

    def __init__(self, model=None, num_vox=128):

        self.num_vox = num_vox

        if model is None:
            self.starting_volume = np.zeros((num_vox, num_vox))
        elif model=='shepp-logan':
            self.starting_volume = shepp_logan(num_vox)

