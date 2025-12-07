# models/nrtl.py

import numpy as np
from .base_model import BaseModel

class NRTLModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.tau = params["tau"]
        self.alpha = params["alpha"]

    def gamma(self, x):
        x = np.array(x, dtype=float)
        eps = 1e-12
        x = np.clip(x, eps, 1.0)

        N = len(x)
        G = np.zeros((N, N))
        tau = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                tau[i, j] = self.tau.get((i+1, j+1), 0.0)
                a = self.alpha.get((i+1, j+1), 0.3)
                G[i, j] = np.exp(-a * tau[i, j])

        ln_gamma = np.zeros(N)

        for i in range(N):
            denom_i = sum(x[j] * G[j, i] for j in range(N))
            denom_i = max(denom_i, eps)

            term1 = sum(
                x[j] * G[j, i] * tau[j, i] /
                max(sum(x[k] * G[k, j] for k in range(N)), eps)
                for j in range(N)
            )

            term2 = sum(
                (x[j] * G[i, j] /
                 max(sum(x[k] * G[k, j] for k in range(N)), eps))
                *
                (tau[i, j] -
                 sum(x[k] * G[k, j] * tau[k, j] for k in range(N)) /
                 max(sum(x[k] * G[k, j] for k in range(N)), eps))
                for j in range(N)
            )

            ln_gamma[i] = term1 + term2

        return np.exp(ln_gamma)
