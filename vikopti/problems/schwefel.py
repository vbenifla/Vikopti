import numpy as np
from ..problem import Problem


class Schwefel(Problem):
    """
    Schwefel function.
    """

    def __init__(self, n=2):
        super().__init__(
            bounds=[[-500] * n, [500] * n],
            name="schwefel",
        )

    def func(self, x):

        # Compute objective
        s = 0
        for i in range(self.n_var):
            s += x[i] * np.sin(np.sqrt(np.abs(x[i])))

        f0 = 418.9829 * self.n_var - s

        return (f0,)
