from vikopti import VIKGA
from vikopti.problems import Sphere


def main(n=2):

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Sphere(n)

    # Run optimization
    run_kwargs = {"n_max": 5000, "d_mins": 0.5}
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    main(6)
