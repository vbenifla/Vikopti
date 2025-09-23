from vikopti import Results


def test():

    # Path to results folder
    path = r"results\beam\run"

    # Make results object
    res = Results(path)

    # Plot results
    res.print()
    res.plot_pop()
    res.plot_const()
    res.plot_obj(display=True)


if __name__ == "__main__":
    test()
