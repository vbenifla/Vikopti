import numpy as np
from vikopti.problem import Problem


class Rosen(Problem):
    """
    Rosenbrock constrained function.
    """

    def __init__(self):
        super().__init__(
            bounds=[[-1, -1], [2, 2]],
            n_con=1,
            name="rosen",
            vars=["x", "y"],
        )

    def func(self, x):

        # Get variables
        x0, x1 = x

        # Compute objective
        f0 = (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2

        # Compute constraints
        g0 = x0**2 + x1**2 - 2

        return (f0, g0)
