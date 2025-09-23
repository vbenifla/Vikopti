from vikopti import VIKGA
from vikopti.problems import Schwefel, GLC, Rosen


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = GLC()

    # Run optimization
    run_kwargs = {
        "n_min": 50,
        "n_max": 5000,
        "n_gen": 300,
        "decimals": 3,
    }
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    main()
