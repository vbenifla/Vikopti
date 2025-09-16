import numpy as np
import matplotlib.pyplot as plt
from .utils import CMAP, BLUE


class Problem:
    """
    Class representing an optimization problem.
    """

    def __init__(self, bounds: list, n_obj: int = 1, n_con: int = 0, **kwargs):
        """
        Initialize the Problem object and set the different attributes.

        Parameters
        ----------
        bounds : list
            Lower and upper bounds of the design space.
        n_obj : int, optional
            number of objectives, by default 1.
        n_con : int, optional
            number of constraints, by default 0.
        **kwargs : dict, optional
            Keyword arguments for additional problem parameters.
        """
        # Set design space
        self.n_var = len(bounds[0])
        self.vars = kwargs.get("vars", [f"x{i}" for i in range(self.n_var)])
        self.bounds = bounds
        self.sf = float(np.linalg.norm(np.array(bounds[1]) - np.array(bounds[0])))

        # Set objective
        self.n_obj = n_obj
        self.objs = kwargs.get("objs", [f"f{i}" for i in range(self.n_obj)])

        # Set constraints
        self.n_con = n_con
        self.consts = kwargs.get("consts", [f"g{i}" for i in range(self.n_con)])
        self.limits = kwargs.get("limits", [0.0] * self.n_con)

        # Set other parameters
        self.name = kwargs.get("name", "ProblemWithNoName")
        self.plotable = kwargs.get("plotable", True)

    def print(self):
        """
        Print summary of the problem in the console.
        """
        print(f"Problem:   {self.name}")
        print(f"N° vars:   {self.n_var}")
        print(f"N° objs:   {self.n_obj}")
        print(f"N° consts: {self.n_con}")

    def plot(self, nx: int = 1000, display=True):
        """
        Plot objective function.

        Parameters
        ----------
        nx : int, optional
            Discretization, by default 1000.
        display : bool, optional
            Flag to display the figure, by default True.

        Returns
        -------
        tuple
            Matplotlib's figure and axes objects.
        """
        # Plot only 1D and 2D objective functions
        if self.n_var < 3:

            # Create figure
            fig, ax = plt.subplots()

            # If 1D objective function
            if self.n_var == 1:

                # Generate domain
                x = np.linspace(self.bounds[0][0], self.bounds[1][0], nx)

                # Evaluate problem
                results = self.func([x])
                f = results[0]
                if self.n_con > 0:
                    gs = results[1:]
                    mask = np.zeros_like(x, dtype=bool)
                    for g in gs:
                        mask |= g > 0

                # Plot objective
                ax.plot(x, f, color=BLUE)

                # Shade the areas where any constraint is satisfied
                if self.n_con:
                    plt.fill_between(
                        x, f, where=mask, color="grey", alpha=0.5, label="g > 0"
                    )

                # Format figure
                ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
                ax.set_xlabel(self.vars[0])
                ax.set_ylabel(self.objs[0], rotation=0)
                if self.n_con:
                    ax.legend()

            # If 2D
            elif self.n_var == 2:

                # Generate domain
                x = np.linspace(self.bounds[0][0], self.bounds[1][0], nx)
                y = np.linspace(self.bounds[0][1], self.bounds[1][1], nx)
                x, y = np.meshgrid(x, y)

                # Evaluate problem
                results = self.func([x, y])
                z = results[0]

                # Mask constrained regions
                if self.n_con > 0:
                    const = results[1:]
                    mask = np.all(
                        const <= np.zeros(self.n_con)[:, np.newaxis, np.newaxis], axis=0
                    )
                    z[~mask] = np.nan

                # Plot
                cs = ax.imshow(
                    z,
                    origin="lower",
                    extent=(
                        self.bounds[0][0],
                        self.bounds[1][0],
                        self.bounds[0][1],
                        self.bounds[1][1],
                    ),
                    cmap=CMAP,
                )

                # Format figure
                ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
                ax.set_ylim(self.bounds[0][1], self.bounds[1][1])
                ax.set_xlabel(self.vars[0])
                ax.set_ylabel(self.vars[1], rotation=0)
                ax.set_aspect("equal")
                cbar = fig.colorbar(cs, ax=ax, pad=0.025)
                cbar.ax.set_ylabel(self.objs[0], rotation=0)

            # Display figure
            if display:
                plt.show()

            return fig, ax
