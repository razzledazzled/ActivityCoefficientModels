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

        # Build tau and G matrices
        tau = np.zeros((N, N))
        G = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                tau[i, j] = self.tau.get((i+1, j+1), 0.0)
                alpha = self.alpha.get((i+1, j+1), 0.3)
                G[i, j] = np.exp(-alpha * tau[i, j])

        # Precompute denominators
        S = np.zeros(N)
        for j in range(N):
            S[j] = max(sum(x[k] * G[k, j] for k in range(N)), eps)

        ln_gamma = np.zeros(N)

        for i in range(N):
            # First summation term
            term1 = sum(
                x[j] * G[j, i] * tau[j, i] / S[i]
                for j in range(N)
            )

            # Second summation term
            term2 = 0.0
            for j in range(N):
                num = sum(x[k] * G[k, j] * tau[k, j] for k in range(N))
                frac = num / S[j]

                term2 += (x[j] * G[i, j] / S[j]) * (tau[i, j] - frac)

            ln_gamma[i] = term1 + term2

        return np.exp(ln_gamma)
