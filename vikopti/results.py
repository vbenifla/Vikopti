import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .fitness import compute_fitness, scale_fitness
from .utils import CMAP, BLUE


class Results:
    """
    Class representing VIKGA run results.
    """

    def __init__(self, save_dir):
        """
        Initializes Results object from a results folder.

        Parameters
        ----------
        save_dir : str
            Path to results folder.
        """
        # Read configuration and problem file
        with open(os.path.join(save_dir, "config.yml"), "r") as file:
            self.config = yaml.safe_load(file)
        with open(os.path.join(save_dir, "pb.yml"), "r") as file:
            self.pb = yaml.safe_load(file)

        # Read algorithm's population and generation data
        self.df_pop = pd.read_csv(os.path.join(save_dir, "pop.csv"))
        self.df_gen = pd.read_csv(os.path.join(save_dir, "gen.csv"))

        # Compute fitness and final results
        x = self.df_pop[self.pb["vars"]].to_numpy()
        obj = self.df_pop[self.pb["objs"]].to_numpy()
        const = self.df_pop[self.pb["consts"]].to_numpy()
        dist = cdist(x, x, "euclidean") / self.pb["sf"]
        f, pen = compute_fitness(obj, const)
        fs, mins = scale_fitness(
            f, pen, dist, d=self.config["d_mins"], n=self.config["n_mins"]
        )
        self.df_pop["f"] = f
        self.df_pop["pen"] = pen
        self.df_pop["fs"] = fs
        self.mins = mins

        # Make local minima DataFrame
        df_mins = self.df_pop.iloc[self.mins].drop(columns=["fs"])
        self.df_mins = df_mins.sort_values(by="f", ascending=False)

    def print(self):
        """
        Print results.
        """
        print("#################### RESULTS ###################")
        print(f"Problem:  {self.pb['name']}")
        print(f"N° gens:  {self.df_gen.shape[0]-1}")
        print(f"N° evals: {self.df_pop.shape[0]}")
        print("Local minima:")
        print(self.df_mins.to_string(index=False, justify="left"))
        print("################################################")

    def plot(self):
        """
        Plot results.
        """
        self.plot_obj()
        self.plot_const()
        self.plot_pop(display=True)

    def plot_pop(self, param="fs", display=False):
        """
        Plot final population.

        Parameters
        ----------
        display : bool, optional
            Flag to display the figure, by default True.

        Returns
        -------
        tuple/grid
            Either Matplotlib's figure and axes objects or Seaborn grid object.
        """
        n_var = len(self.pb["vars"])
        if n_var == 1:

            # Create figure
            fig, ax = plt.subplots()

            # Scatter plot
            x = self.df_pop[self.pb["vars"][0]]
            y = self.df_pop[param]
            scatter = ax.scatter(x, y, c=self.df_pop[param], cmap=CMAP, s=10)

            # Set labels
            ax.set_xlabel(self.pb["vars"])
            ax.set_ylabel("f", rotation=0)

            # Display figure
            fig.tight_layout()
            if display:
                plt.show()

            return fig, ax

        if n_var == 2:

            # Create figure
            fig, ax = plt.subplots()

            # Scatter plot
            x = self.df_pop[self.pb["vars"][0]]
            y = self.df_pop[self.pb["vars"][1]]
            scatter = ax.scatter(x, y, c=self.df_pop[param], cmap=CMAP, s=10)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label(param, rotation=0, labelpad=15)
            cbar.set_ticks(
                np.linspace(self.df_pop[param].min(), self.df_pop[param].max(), 5)
            )

            # Set labels
            ax.set_xlabel(self.pb["vars"][0])
            ax.set_ylabel(self.pb["vars"][1], rotation=0)

            # Display figure
            fig.tight_layout()
            if display:
                plt.show()

            return fig, ax

        if n_var > 2:

            # Make seaborn PairGrid
            df = self.df_pop.sort_values("f", ascending=True)
            g = sns.PairGrid(
                data=df,
                vars=self.pb["vars"],
                diag_sharey=False,
                corner=True,
                despine=True,
                layout_pad=0,
            )

            # Pairwise scatter plot of the population with scaled fitness as colorbar
            g.map_lower(
                sns.scatterplot,
                hue=df[param],
                palette=CMAP,
                size=df[param],
                sizes=(5, 20),
            )

            # Pairwise contour plot of the population
            g.map_upper(sns.kdeplot, fill=True, cut=0)

            # Density plot of each variables
            g.map_diag(sns.kdeplot, fill=True, cut=0)

            # Manually overlay minima
            for id_min in self.mins:
                for i, j in zip(*np.tril_indices(len(self.pb["vars"]), -1)):
                    ax = g.axes[i, j]
                    ax.scatter(
                        self.df_pop.iloc[id_min][self.pb["vars"][j]],
                        self.df_pop.iloc[id_min][self.pb["vars"][i]],
                        marker="s",
                        facecolors="None",
                        edgecolors="k",
                        linewidths=1,
                        s=100,
                        label="optima",
                        zorder=10,
                    )

                    # Set axis limits to match domain bounds
                    ax.set_xlim(self.pb["bounds"][0][j], self.pb["bounds"][1][j])
                    ax.set_ylim(self.pb["bounds"][0][i], self.pb["bounds"][1][i])

            # Set axis limits on diagonal plots
            for i, ax in enumerate(np.diag(g.axes)):
                ax.set_xlim(self.pb["bounds"][0][i], self.pb["bounds"][1][i])

            # Add scaled fitness colorbar
            norm = plt.Normalize(0, 1)
            sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
            sm.set_array([])

            # Create a new axis for the colorbar
            cbar_ax = g.figure.add_axes(
                [0.93, 0.2, 0.02, 0.6]
            )  # [left, bottom, width, height]
            cbar = g.figure.colorbar(sm, cax=cbar_ax)
            cbar.set_label(
                param, rotation=0, labelpad=15
            )  # Horizontal label with padding
            cbar.set_ticks(np.linspace(0, 1, 5))  # Set ticks from 0 to 1

            # Display figure
            plt.tight_layout()
            if display:
                plt.show()

            return g

    def plot_obj(self, id_obj=0, display=False):
        """
        Plot best objective evolution.

        Parameters
        ----------
        id_obj : int, optional
            objective id, by default 0.
        display : bool, optional
            Flag to display the figure, by default False.

        Returns
        -------
        tuple
            Matplotlib's figure and axes objects
        """
        # Create figure
        fig, ax = plt.subplots()

        # Objective evolution
        x = self.df_gen["c_pop"]
        y = self.df_pop.iloc[self.df_gen["min_idx"]][self.pb["objs"][id_obj]]
        ax.plot(
            x,
            y,
            linestyle="-",
            marker=".",
            markersize=6,
            markerfacecolor="none",
            markevery=int(self.df_gen.shape[0] / 10),
            c=BLUE,
        )

        # Format plot
        ax.set_ylabel(self.pb["objs"][id_obj], rotation=0)
        ax.set_xlabel("evaluations")

        # Display
        fig.tight_layout()
        if display:
            plt.show()

        return fig, ax

    def plot_const(self, display=False):
        """
        Plot best constraints evolution.

        Parameters
        ----------
        display : bool, optional
            Flag to display the figure, by default True.

        Returns
        -------
        tuple
            Matplotlib's figure and axes objects.
        """
        # Get evaluations
        x = self.df_gen["c_pop"]

        # Plot constraints
        n_con = len(self.pb["consts"])
        if n_con > 0:

            # Create subplots
            n = int(np.ceil(np.sqrt(n_con)))
            fig, axes = plt.subplots(nrows=n, ncols=n, squeeze=False)
            axes = axes.flatten()

            # Plot each constraint
            for i in range(n_con):
                ax = axes[i]
                ax.axhline(0, c="gray", linestyle="--")
                y = self.df_pop.iloc[self.df_gen["min_idx"]][self.pb["consts"][i]]
                ax.plot(
                    x,
                    y,
                    linestyle="-",
                    marker=".",
                    markersize=6,
                    markerfacecolor="none",
                    markevery=int(self.df_gen.shape[0] / 10),
                    c=BLUE,
                )
                ax.set_title(self.pb["consts"][i])

                # Only set x-label for the bottom row
                if i >= (n * (n - 1)):
                    ax.set_xlabel("Evaluations")

            # Hide any empty subplots
            for j in range(n_con, len(axes)):
                fig.delaxes(axes[j])

            # Display
            fig.tight_layout()
            if display:
                plt.show()

            return fig, ax
