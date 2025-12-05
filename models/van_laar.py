# models/van_laar.py

import numpy as np
from models.base_model import BaseModel

class VanLaarModel(BaseModel):
    """
    Van Laar model for activity coefficients.
    Parameters:
        A: dictionary of pairwise Van Laar parameters, keys are (i,j) tuples
           for N components. Example: {(1,2): 1.2, (2,1): 0.9, ...}
    """

    def gamma(self, x, epsilon=1e-12):
        """
        Compute activity coefficients for any number of components safely.
        
        x: list or array of mole fractions, length N
        epsilon: small value to avoid division by zero
        
        Returns: array of gamma values
        """
        x = np.array(x)
        N = len(x)
        A = self.params["A"]  # {(i,j): value} with integer tuples

        gamma = np.zeros(N)

        for i in range(N):
            sum_terms = 0.0

            for j in range(N):
                if j == i:
                    continue

                # Access pairwise parameters
                try:
                    Aij = A[(i+1, j+1)]
                    Aji = A[(j+1, i+1)]
                except KeyError:
                    raise KeyError(f"Missing Van Laar parameter for pair ({i+1},{j+1})")

                # Compute denominator safely
                denom = sum(A[(k+1, j+1)] * x[k] for k in range(N) if k != j)
                if denom < epsilon:
                    denom = epsilon  # prevent division by zero

                # Add term to gamma sum
                sum_terms += Aij * ((Aji * x[j]) / denom) ** 2

            # Compute gamma[i]
            gamma[i] = np.exp(sum_terms)

        return gamma
