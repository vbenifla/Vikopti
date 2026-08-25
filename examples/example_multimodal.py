from vikopti import VIKGA, Results
from vikopti.problems import Schwefel, GLC, Rosen


def main():

    # Initialize algorithm and problem
    algo = VIKGA()
    pb = Rosen()

    # Run optimization
    run_kwargs = {"n_min": 50, "n_gen": 1000}
    algo.run(pb, **run_kwargs)

    # Make results object
    res = Results(algo.config.save_dir)

    # Plot some results
    res.plot_pop()
    res.plot_const()
    res.plot_obj(display=True)


if __name__ == "__main__":
    main()
