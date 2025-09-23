import os
from vikopti import VIKGA
from vikopti.problems import Beam


def test():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Beam()

    # Set run options
    run_kwargs = {
        "n_min": 50,
        "n_max": 1000,
        "sample": "sobol",
        "crossover": "pnx",
        "mutation": "um",
        "pm": 1,
        "n_mins": 10,
        "d_mins": 0.15,
        "decimals": 3,
        "save_dir": "test",
    }

    # Run optimization
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    test()
