import os
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .operators import compute_fitness, scale_fitness
from .utils import CMAP, BLUE


class Results:
    """
    Class representing VIKGA run results.
    """

    def __init__(self, summary, pop, gen):
        """
        Constructs the Results object and initializes attributes.

        Parameters
        ----------
        summary : dict
            Dictionary with the algorithm's summary (config, pb, res).
        pop : np.ndarray
            Population's design variables, objectives and constraints values.
        gen : list
            List of results per generation.
        """
        # Set algorithm's configuration, problem and results
        self.config = summary["config"]
        self.pb = summary["pb"]
        self.res = summary["res"]

        # Make generation DataFrame
        cols = ["c_pop", "c_cross", "c_mute", "n_mins", "min"]
        df_gen = pd.DataFrame(gen, columns=cols)
        self.df_gen = df_gen
        self.df_gen["c_save"] = (self.config["n_cross"] * 2 + self.config["n_mute"]) - (
            self.df_gen["c_cross"] + self.df_gen["c_mute"]
        )

        # Make population DataFrame
        cols = self.pb["vars"] + self.pb["objs"] + self.pb["consts"] + ["f", "fs"]
        df_pop = pd.DataFrame(pop, columns=cols)
        self.df_pop = df_pop

        # Make local minima DataFrame
        df_mins = self.df_pop.iloc[self.res["mins"]].drop(columns=["fs"])
        nvar = len(self.pb["vars"])
        x = pop[:, :nvar]
        diff = x[self.res["mins"]][:, None, :] - x[self.res["mins"]][None, :, :]
        dist = np.linalg.norm(diff, axis=-1) / self.pb["sf"]
        df_mins["dist"] = np.min(np.where(np.eye(dist.shape[0], dtype=bool), np.inf, dist), axis=0)
        self.df_mins = df_mins.sort_values(by="f", ascending=False)

    @classmethod
    def from_path(cls, path):
        """
        Initializes Results object from a results folder.

        Parameters
        ----------
        path : str
            Path to results folder.

        Returns
        -------
        Results
            Results object.
        """
        # Read summary file
        with open(os.path.join(path, "sum.yml"), "r") as file:
            summary = yaml.safe_load(file)
        config = summary["config"]
        pb = summary["pb"]

        # Read algorithm's population data
        pop = np.loadtxt(os.path.join(path, "pop.txt"))
        n_var = len(pb["vars"])
        n_obj = len(pb["objs"])
        n_con = len(pb["consts"])
        x = pop[:, :n_var]
        obj = pop[:, n_var : n_var + n_obj]
        const = pop[:, n_var + n_obj : n_var + n_obj + n_con]
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) / pb["sf"]
        f, is_feasible, pen = compute_fitness(obj[:, 0], const, alpha=config["alpha"])
        fs, mins = scale_fitness(f, dist, is_feasible, d=config["d_mins"], n=config["n_mins"])
        pop = np.concatenate([x, obj, const, f[:, np.newaxis], fs[:, np.newaxis]], axis=1)

        # Read algorithm generation results
        gen = np.loadtxt(os.path.join(path, "gen.txt"))

        # Try reading results
        if "res" not in summary:
            default_res = {
                "run_time": 0.0,
                "n_eval": len(pop),
                "mins": mins,
                "x": list(x[mins[0]]),
                "obj": (obj[mins[0]]),
                "const": (const[mins[0]]),
            }
            summary["res"] = default_res

        return cls(summary, pop, gen)

    def print(self):
        """
        Print results.
        """
        print("#################### RESULTS ###################")
        print(f"Problem:  {self.pb['name']}")
        print(f"N° gens:  {self.df_gen.shape[0]-1}")
        print(f"N° evals: {self.res['n_eval']}")
        print(f"N° saves: {self.df_gen['c_save'].sum()}")
        print(f"Run time: {self.res['run_time']} s")
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

    def plot_pop(self, display=False):
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
            y = self.df_pop["f"]
            scatter = ax.scatter(x, y, c=self.df_pop["f"], cmap=CMAP, s=10)

            # Set labels
            ax.set_xlabel(self.pb["vars"])
            ax.set_ylabel("f", rotation=0)

            # Display figure
            if display:
                plt.show()

            return fig, ax

        if n_var == 2:

            # Create figure
            fig, ax = plt.subplots()

            # Scatter plot
            x = self.df_pop[self.pb["vars"][0]]
            y = self.df_pop[self.pb["vars"][1]]
            scatter = ax.scatter(x, y, c=self.df_pop["f"], cmap=CMAP, s=10)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("f", rotation=0, labelpad=15)
            cbar.set_ticks(np.linspace(self.df_pop["f"].min(), self.df_pop["f"].max(), 5))

            # Set labels
            ax.set_xlabel(self.pb["vars"][0])
            ax.set_ylabel(self.pb["vars"][1], rotation=0)

            # Display figure
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
                # corner=True,
                # despine=True,
                layout_pad=0,
            )

            # Pairwise scatter plot of the population with scaled fitness as colorbar
            g.map_lower(
                sns.scatterplot,
                hue=df["f"],
                palette=CMAP,
                size=df["f"],
                sizes=(20, 10),
            )

            # Pairwise contour plot of the population
            g.map_upper(sns.kdeplot, fill=True, cut=0)

            # Density plot of each variables
            g.map_diag(sns.kdeplot, fill=True, cut=0)

            # Manually overlay minima
            for id_min in self.res["mins"]:
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
            norm = plt.Normalize(df["f"].min(), df["f"].max())
            sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
            sm.set_array([])

            # Create a new axis for the colorbar
            cbar_ax = g.figure.add_axes([0.93, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
            cbar = g.figure.colorbar(sm, cax=cbar_ax)
            cbar.set_label("f", rotation=0, labelpad=15)  # Horizontal label with padding
            cbar.set_ticks(np.linspace(df["f"].min(), df["f"].max(), 5))  # Set ticks from 0 to 1

            # Display figure
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
        y = self.df_pop.iloc[self.df_gen["min"]][self.pb["objs"][id_obj]]
        ax.plot(
            x,
            y,
            linestyle="-",
            marker=".",
            markersize=6,
            markerfacecolor="none",
            markevery=25,
            c=BLUE,
        )

        # Format plot
        ax.set_ylabel(self.pb["objs"][id_obj], rotation=0)
        ax.set_xlabel("evaluations")

        # Display
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
                y = self.df_pop.iloc[self.df_gen["min"]][self.pb["consts"][i]]
                ax.plot(
                    x,
                    y,
                    linestyle="-",
                    marker=".",
                    markersize=6,
                    markerfacecolor="none",
                    markevery=25,
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
            if display:
                plt.show()

            return fig, ax
