from abc import ABC, abstractmethod
import numpy as np


class BatchTransform(ABC):
    ''' Abstact base class to signify that a transform callable supports processing multiple data points in batch,
        i.e. can handle 5D [batch, fundamental_angle, orthogonal_angle, radius, hrir_idx] arrays in addition to 
        4D [fundamental_angle, orthogonal_angle, radius, hrir_idx] arrays.'''

    @abstractmethod
    def __call__(self, values: np.ma.MaskedArray):
        pass