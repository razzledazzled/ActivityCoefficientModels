# models/wilson.py

import numpy as np
from .base_model import BaseModel

class WilsonModel(BaseModel):
    def __init__(self, params):
        """
        params: {'Lambda': {(i,j): value}}
        Keys already converted to tuples by load_params()
        """
        super().__init__(params)
        self.Lambda = params["Lambda"]  # no splitting needed

    def gamma(self, x):
        x = np.array(x, dtype=float)
        eps = 1e-12
        x = np.clip(x, eps, 1.0)
        x /= np.sum(x)

        N = len(x)
        gamma = np.zeros(N)

        # Precompute denominators for efficiency
        denom = np.zeros((N,))
        for j in range(N):
            denom[j] = max(sum(x[k] * self.Lambda.get((k+1,j+1), 1.0) for k in range(N)), eps)

        for i in range(N):
            sum_xL_ji = max(sum(x[j] * self.Lambda.get((j+1, i+1), 1.0) for j in range(N)), eps)
            second_term = sum(x[j] * self.Lambda.get((i+1, j+1), 1.0) / denom[j] for j in range(N))
            gamma[i] = np.exp(1 - np.log(sum_xL_ji) - second_term)

        return gamma

