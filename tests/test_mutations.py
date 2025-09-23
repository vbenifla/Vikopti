import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vikopti.utils import CMAP
from vikopti.operators import mutate


def test(n_var=2, method="pm", random=False):

    # Set design space
    xvars = [f"x{i}" for i in range(n_var)]
    bounds = [[-1.0] * n_var, [1.0] * n_var]
    lower, upper = np.array(bounds[0]), np.array(bounds[1])

    # Generate 2 parents
    if random:
        x = np.random.uniform(lower, upper)
    else:
        x = np.full(n_var, 0.0)

    # Prepare offspring generation
    n_mute = 50
    n_eta = 10
    etas = np.linspace(1, 25, n_eta)
    data = []

    # Add parent
    data.append(list(x) + [etas[0], "parent"])

    # Generate offspring for each eta value
    for eta in etas:
        for _ in range(n_mute):
            xoff = mutate(x, bounds, method=method, eta=eta, pm=0.9)
            data.append(list(xoff) + [eta, "offspring"])

    # Create DataFrame
    cols = xvars + ["eta", "type"]
    df = pd.DataFrame(data, columns=cols)

    # PairGrid with only lower triangle
    g = sns.PairGrid(
        data=df[df["type"] == "offspring"],
        vars=xvars,
        diag_sharey=False,
        layout_pad=0,
        corner=True,
    )

    # Pairwise scatter plot of the offspring colored by eta
    g.map_lower(
        sns.scatterplot,
        hue=df[df["type"] == "offspring"]["eta"],
        palette=CMAP,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Diagonal KDE plots of each variable
    g.map_diag(sns.kdeplot, fill=True, cut=0)

    # Manually overlay parent points on lower triangle
    for i, j in zip(*np.tril_indices(len(xvars), -1)):
        ax = g.axes[i, j]
        ax.scatter(
            df[df["type"] == "parent"][xvars[j]],
            df[df["type"] == "parent"][xvars[i]],
            marker="*",
            s=200,
            c="k",
            label="parent",
            zorder=10,
        )

        # Set axis limits to match domain bounds
        ax.set_xlim(bounds[0][j], bounds[1][j])
        ax.set_ylim(bounds[0][i], bounds[1][i])

    # Set axis limits on diagonal plots
    for i, ax in enumerate(np.diag(g.axes)):
        ax.set_xlim(bounds[0][i], bounds[1][i])

    # Add colorbar for eta
    norm = plt.Normalize(
        df[df["type"] == "offspring"]["eta"].min(),
        df[df["type"] == "offspring"]["eta"].max(),
    )
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    # Add colorbar axis
    cbar_ax = g.figure.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = g.figure.colorbar(sm, cax=cbar_ax)
    cbar.set_label("eta", rotation=0, labelpad=15)
    cbar.set_ticks(
        np.linspace(
            df[df["type"] == "offspring"]["eta"].min(),
            df[df["type"] == "offspring"]["eta"].max(),
            5,
        )
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test(method="um")
    test(method="nm", random=True)
    test(5, "pm", True)
