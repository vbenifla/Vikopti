import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vikopti.operators import sample


def test(n_var=2):

    # Set design space
    xvars = [f"x{i}" for i in range(n_var)]
    bounds = [[-1.0] * n_var, [10.0] * n_var]

    # Generate samples
    n = 32
    data = []
    for method in ["random", "lhs", "sobol"]:
        xs = sample(n, bounds, method=method)
        print(len(xs))
        for x in xs:
            data.append(list(x) + [method])

    # Create DataFrame
    cols = xvars + ["method"]
    df = pd.DataFrame(data, columns=cols)

    # Make seaborn PairGrid
    g = sns.PairGrid(
        data=df,
        vars=xvars,
        hue="method",
        diag_sharey=False,
        corner=True,
        despine=True,
        layout_pad=0,
    )

    # Scatter plot on the lower diag
    g.map_lower(
        sns.scatterplot,
    )
    # Density plot on the diag
    g.map_diag(sns.kdeplot, fill=True, cut=0)

    # Set axis limits to match domain bounds
    for i, j in zip(*np.tril_indices(len(xvars), -1)):
        ax = g.axes[i, j]
        ax.set_xlim(bounds[0][j], bounds[1][j])
        ax.set_ylim(bounds[0][i], bounds[1][i])

    # Set axis limits on diagonal plots
    for i, ax in enumerate(np.diag(g.axes)):
        ax.set_xlim(bounds[0][i], bounds[1][i])

    # Show plot
    g.add_legend()
    plt.show()


if __name__ == "__main__":
    test(5)
