from vikopti import VIKGA
from vikopti.problems import Beam


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Beam()

    # Run optimization
    run_kwargs = {"n_max": 5000, "n_gen": 2000, "d_mins":0.2}

    # Run optimziations
    algo.run(pb, save_dir="test", **run_kwargs)
    # algo.run_multiple(pb, 3, save_dir="validation", **run_kwargs)


if __name__ == "__main__":
    main()
