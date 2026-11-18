from abc import ABC, abstractmethod
import numpy as np


class BatchTransform(ABC):
    ''' Abstract base class for transforms that do not impose any restrictions on the values they can process,
        i.e. can dense and masked NumPy arrays of any dimensionality.'''

    @abstractmethod
    def __call__(self, values: np.ndarray):
        pass
