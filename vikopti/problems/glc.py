import numpy as np
from vikopti.problem import Problem


class GLC(Problem):
    """
    Gomez-Levy constrained function.
    """

    def __init__(self):
        super().__init__(
            bounds=[[-1, -1], [1, 1]],
            n_con=1,
            name="glc",
            vars=["x", "y"],
        )

    def func(self, x):

        # Get variables
        x0, x1 = x

        # Compute objective
        f0 = 4.0 * x0**2 - 2.1 * x0**4 + (x0**6) / 3.0 + x0 * x1 - 4.0 * x1**2 + 4 * x1**4

        # Compute constraints
        g0 = 2.0 * np.sin(2.0 * np.pi * x1) ** 2 - np.sin(4.0 * np.pi * x0) - 1.5

        return (f0, g0)
