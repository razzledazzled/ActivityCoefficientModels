import numpy as np
from models.base_model import BaseModel

class VanLaarDifferentialModel(BaseModel):
    """
    Van Laar activity coefficient model (differential form) for N-component systems.
    Parameters:
        A : dict of pairwise Van Laar parameters, keys are tuples (i,j) with 1-based indexing
            Example for 3 components:
            A = {
                (1,2): 1.2, (2,1): 0.9,
                (1,3): 1.5, (3,1): 1.1,
                (2,3): 0.8, (3,2): 1.0
            }
    """

    def gamma(self, x, eps=1e-12):
        x = np.array(x, dtype=float)
        N = len(x)
        if N < 2:
            raise ValueError("Van Laar model requires at least 2 components.")

        # Clip mole fractions to avoid division by zero
        x = np.clip(x, eps, 1.0)

        ln_gamma = np.zeros(N)

        for i in range(N):
            # Compute numerator and denominator for component i
            denom = x[i]  # start with xi
            numerator = 0.0

            for j in range(N):
                if j == i:
                    continue
                r_ji = self.params["A"][(j+1, i+1)] / self.params["A"][(i+1, j+1)]
                denom += x[j] * r_ji

            denom_sq = denom**2

            # Numerator: sum over all other components
            for j in range(N):
                if j == i:
                    continue
                A_ij = self.params["A"][(i+1, j+1)]
                r_ji = self.params["A"][(j+1, i+1)] / A_ij
                term = x[j]**2 * A_ij * r_ji**2
                numerator += term

                # Cross terms with k>j
                for k in range(N):
                    if k == i or k == j:
                        continue
                    A_ik = self.params["A"][(i+1, k+1)]
                    r_ki = self.params["A"][(k+1, i+1)] / A_ik
                    A_jk = self.params["A"][(j+1, k+1)]
                    r_jk = self.params["A"][(k+1, j+1)] / A_jk
                    cross = x[j]*x[k]*r_ji*r_ki*(A_ij + A_ik - A_jk*(A_ik/self.params["A"][(k+1,j+1)]))
                    numerator += cross

            ln_gamma[i] = numerator / denom_sq

        return np.exp(ln_gamma)
