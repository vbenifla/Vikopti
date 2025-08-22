import numpy as np
from vikopti import VIKGA
from vikopti.problem import Problem


# Simple optimization problem definition
class Simple(Problem):
    def __init__(self, constrained=False):
        super().__init__(
            bounds=[[-1], [1]],
            n_con=1 if constrained else 0,
            name="simple",
        )

    def func(self, x):

        # Get variables
        x0 = x[0]

        # Some parameters
        X = np.array([-0.2, 0.4, 0.8])
        H = np.array([0.8, 0.4, 0.6])
        W = np.array([300, 80, 100])

        # Compute objective
        f0 = 0.0
        for i in range(len(X)):
            f0 += H[i] * np.exp(-W[i] * (x0 - X[i]) ** 2)

        # compute constraints
        g0 = -np.sin(10 * x0)
        # g0 = x0 - 0.5
        g0 = 0.8 - abs(x0 - 0.4) / 0.4

        if self.n_con > 0:
            return (-f0, g0)
        else:
            return (-f0,)


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Simple(False)

    # Run optimization
    run_kwargs = {
        "n_min": 10,
        "n_max": 100,
        "n_mins": 4,
        "n_crowd": 4,
    }
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    main()
