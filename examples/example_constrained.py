from vikopti import VIKGA
from vikopti.problems import Beam


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Beam()

    # Run optimization
    run_kwargs = {
        "n_min": 100,
        "n_max": 5000,
        "n_gen": 500,
        "decimals": 3,
    }
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    main()
