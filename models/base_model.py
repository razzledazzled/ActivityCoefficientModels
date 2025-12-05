# models/base_model.py

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all activity coefficient models.
    All models must implement the gamma() method.
    """

    def __init__(self, params):
        """
        params: dictionary containing binary interaction parameters
        """
        self.params = params

    @abstractmethod
    def gamma(self, x):
        """
        Compute activity coefficients Y_i for a mixture.
        x is an array-like composition vector [x1, x2, x3].
        Returns array of Y_i.
        """
        pass
