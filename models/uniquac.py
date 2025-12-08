# models/uniquac.py

import numpy as np
from .base_model import BaseModel


class UNIQUACModel(BaseModel):
    def __init__(self, params):
        """
        params should contain:
          r : dictionary of van der Waals volume parameters
          q : dictionary of surface area parameters
          a : dictionary of binary interaction energies
        """
        super().__init__(params)

        self.r = params["r"]      # size parameters
        self.q = params["q"]      # surface parameters
        self.a = params["a"]      # interaction parameters
        self.z = 10               # coordination number (fixed for UNIQUAC)

    def gamma(self, x):
        """
        Calculate activity coefficients using the UNIQUAC model.
        x = list or array of mole fractions (should sum to 1)
        """

        # ---------- Basic setup ----------
        x = np.array(x, dtype=float)
        N = len(x)

        # Small value to prevent numerical issues like log(0)
        eps = 1e-12

        # Extract pure component parameters in the correct order
        # r = molecular volume parameter for each component
        # q = molecular surface area parameter for each component
        r = np.array([self.r[i+1] for i in range(N)], dtype=float)
        q = np.array([self.q[i+1] for i in range(N)], dtype=float)

        # Prevent zero or negative mole fractions (stability trick)
        x = np.clip(x, eps, 1.0)

        # Optional: renormalize so x still sums to 1
        x = x / np.sum(x)


        # ---------- Combinatorial part ----------
        # This part accounts for:
        # - molecular size differences
        # - shape differences
        #
        # phi  = volume fraction
        # theta = surface area fraction

        phi = r * x / np.sum(r * x)
        theta = q * x / np.sum(q * x)

        # l parameter (shape factor correction)
        l = (self.z / 2) * (r - q) - (r - 1)

        # Combinatorial contribution to ln(gamma)
        # Describes entropy (packing/size effects)
        ln_gamma_comb = (
            np.log(phi / x) +
            (self.z / 2) * q * np.log(theta / phi) +
            l -
            (phi / x) * np.sum(x * l)
        )


        # ---------- Residual part ----------
        # This part accounts for:
        # - energetic interactions between unlike molecules

        # Build tau matrix (interaction strength)
        # tau_ij = exp( -a_ij /(RT) ), but we omit T here for now
        tau = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                aij = self.a.get((i+1, j+1), 0.0)
                tau[i, j] = np.exp(-aij)

        # Precompute denominators:
        # denom_j = sum_k(theta_k * tau_kj)
        denom = np.zeros(N)
        for j in range(N):
            denom[j] = max(sum(theta[k] * tau[k, j] for k in range(N)), eps)

        # Residual contribution to ln(gamma)
        ln_gamma_res = np.zeros(N)
        for i in range(N):

            # First term: log of local interaction sum
            sum1 = max(sum(theta[j] * tau[j, i] for j in range(N)), eps)

            # Second term: local composition correction
            sum2 = 0.0
            for j in range(N):
                sum2 += theta[j] * tau[i, j] / denom[j]

            # Full residual term
            ln_gamma_res[i] = q[i] * (1 - np.log(sum1) - sum2)


        # ---------- Final activity coefficients ----------
        # ln(gamma) = combinatorial + residual
        ln_gamma = ln_gamma_comb + ln_gamma_res

        # Convert from ln(gamma) to gamma
        return np.exp(ln_gamma)
