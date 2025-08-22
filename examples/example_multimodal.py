from vikopti import VIKGA
from vikopti.problems import Schwefel, GLC, Rosen


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Schwefel()

    # Run optimization
    algo.run(pb)


if __name__ == "__main__":
    main()
