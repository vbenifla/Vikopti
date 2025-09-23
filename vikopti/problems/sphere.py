import numpy as np
from ..problem import Problem


class Sphere(Problem):
    """
    Sphere function.
    """

    def __init__(self, n=2):
        super().__init__(
            bounds=[[-1] * n, [1] * n],
            name="sphere",
        )

    def func(self, x):

        # Compute objective
        f0 = 0.0
        for i in range(self.n_var):
            f0 += x[i] ** 2

        return (f0,)
