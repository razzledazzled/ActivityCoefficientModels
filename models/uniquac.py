# models/uniquac.py

import numpy as np
from .base_model import BaseModel

class UNIQUACModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        self.r = params["r"]
        self.q = params["q"]
        self.a = params["a"]
        self.z = 10

    def gamma(self, x):
        x = np.array(x, dtype=float)
        N = len(x)
        eps = 1e-12   # numerical safety

        r = np.array([self.r[i+1] for i in range(N)], dtype=float)
        q = np.array([self.q[i+1] for i in range(N)], dtype=float)

        # Prevent zero mole fractions
        x = np.clip(x, eps, 1.0)

        # Combinatorial part
        phi = r * x / np.sum(r * x)
        theta = q * x / np.sum(q * x)

        phi = np.clip(phi, eps, 1.0)
        theta = np.clip(theta, eps, 1.0)

        l = (self.z / 2) * (r - q) - (r - 1)

        ln_gamma_comb = (
            np.log(phi / x) +
            (self.z / 2) * q * np.log(theta / phi) +
            l - (phi / x) * np.sum(x * l)
        )

        # Residual part
        tau = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                aij = self.a.get((i+1, j+1), 0.0)
                tau[i, j] = np.exp(-aij)

        ln_gamma_res = np.zeros(N)
        for i in range(N):
            sum1 = sum(theta[j] * tau[j, i] for j in range(N))
            sum1 = max(sum1, eps)  # protect log

            sum2 = 0.0
            for j in range(N):
                denom = sum(theta[k] * tau[k, j] for k in range(N))
                denom = max(denom, eps)
                sum2 += theta[j] * tau[i, j] / denom

            ln_gamma_res[i] = q[i] * (1 - np.log(sum1) - sum2)

        ln_gamma = ln_gamma_comb + ln_gamma_res
        return np.exp(ln_gamma)
