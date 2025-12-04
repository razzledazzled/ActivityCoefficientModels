# models/van_laar.py

import numpy as np
from models.base_model import BaseModel

class VanLaarModel(BaseModel):
    """
    Van Laar model for activity coefficients.
    Parameters must contain A_ij values for all pairs in a ternary mixture.
    """

    def gamma(self, x):
        x = np.array(x)
        n = len(x)
        A = self.params["A"]   # A is a dictionary {("1","2"): value, ...}

        ln_gamma = np.zeros(n)

        for i in range(n):
            numerator_terms = []
            denom_terms = []

            for k in range(n):
                if k != i:
                    Aik = A[(str(i+1), str(k+1))]
                    denom_terms.append(Aik * x[k])

            denom = sum(denom_terms)

            # sum over j != i
            for j in range(n):
                if j != i:
                    Aij = A[(str(i+1), str(j+1))]
                    Aji = A[(str(j+1), str(i+1))]

                    term = Aij * (Aji * x[j] / denom) ** 2
                    ln_gamma[i] += term

        return np.exp(ln_gamma)
