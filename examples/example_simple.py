from vikopti import VIKGA
from vikopti.problems import F1


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = F1(False)

    # Run optimization
    run_kwargs = {
        "n_min": 20,
        "n_max": 100,
        "n_mins": 4,
    }
    algo.run(pb, **run_kwargs)


if __name__ == "__main__":
    main()
