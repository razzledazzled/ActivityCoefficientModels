# models/van_laar.py

import numpy as np
from .base_model import BaseModel

class VanLaarModel(BaseModel):
    """
    Van Laar model for activity coefficients.
    Parameters:
        A: dictionary of pairwise Van Laar parameters, keys are (i,j) tuples
           for N components. Example: {(1,2): 1.2, (2,1): 0.9, ...}
    """

    def gamma(self, x, epsilon=1e-12):
        """
        Compute activity coefficients for N-component mixtures using Van Laar model.
        
        x: list or array of mole fractions, length N
        epsilon: small value to avoid division by zero
        
        Returns: array of gamma values
        """
        x = np.array(x, dtype=float)
        x /= np.sum(x)  # ensure mole fractions sum to 1
        N = len(x)
        A = self.params["A"]  # {(i,j): value} with integer tuples

        ln_gamma = np.zeros(N)

        for i in range(N):
            sum_terms = 0.0

            for j in range(N):
                if j == i:
                    continue  # skip self-interaction

                # pairwise Van Laar parameters
                try:
                    Aij = A[(i+1, j+1)]
                    Aji = A[(j+1, i+1)]
                except KeyError:
                    raise KeyError(f"Missing Van Laar parameter for pair ({i+1},{j+1})")

                # denominator sum over k != j
                denom = sum(A.get((k+1, j+1), 0.0) * x[k] for k in range(N) if k != j)
                denom = max(denom, epsilon)  # avoid division by zero

                sum_terms += Aij * (Aji * x[j] / denom)

            ln_gamma[i] = sum_terms

        return np.exp(ln_gamma)
