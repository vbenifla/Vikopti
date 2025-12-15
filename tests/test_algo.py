import os
from vikopti import VIKGA
from vikopti.problems import Rosen
from vikopti.problems import Beam


def test():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Beam()

    # Set run options
    run_kwargs = {
        "n_min": 30,
        "n_max": 1000,
        "sample": "lhs",
        "crossover": "pnx",
        "mutation": "um",
        "pm": 1,
        "save_dir": "test",
    }

    # Run optimization
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    test()
