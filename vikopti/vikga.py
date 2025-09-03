import os
import time
import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed, wrap_non_picklable_objects
from .config import Config
from .problem import Problem
from .results import Results
from .operators import sample, compute_fitness, scale_fitness, cross, mutate


class VIKGA:
    """
    Class representing a pretty cool genetic algorithm:
        - It is inspired by the Cumulative Multi-Niching Genetic Algorithm from M. Hall. 2012.
        - The population is cumulative and extends through the generation.
        - It can identify multiple local minima using a fitness scaling operation.
        - It handles constraints via an efficient penalty method.
        - It presents different sampling methods for the initial population.
        - It combines fitness and distance-proportionate selection.
        - It offers multiple crossover and mutation operators.
        - It avoid redundant objective function evaluations.
    """

    def __init__(self):
        """
        Initialize the VIKGA object and set its default configuration.
        """
        # Initialize default configuration
        self.config = Config()

    def run(self, problem: Problem, **run_kwargs):
        """
        Run the algorithm for a given optimization problem.

        Parameters
        ----------
        problem : Problem
            The optimization problem to solve.
        **run_kwargs : dict, optional
            Keyword arguments to override the algorithm's configuration parameters.
            Only keys matching attributes of "Config" are applied.
            See the "Config" class for available parameters and their defaults.

        Returns
        -------
        Results
            Optimization results.
        """
        # Start the optimization process
        self._start(problem, **run_kwargs)

        # Initialize population
        self._initialize()

        # Make population evolve
        while self._evolve():

            # Compute current population fitness
            self._fitness()

            # Perform crossover, mutation and addition operation
            self._crossover()
            self._mutation()
            self._addition()
            self.c_gen += 1

        # Compute final population fitness
        self._fitness()

        # Stop optimization process
        self._stop()

        return self.results

    def _start(self, problem: Problem, **run_kwargs):
        """
        Start the optimization process.

        Parameters
        ----------
        problem : Problem
            The optimization problem to solve.
        **run_kwargs : dict, optional
            Keyword arguments to override the algorithm's configuration parameters.
            Only keys matching attributes of "Config" are applied.
            See the "Config" class for available parameters and their defaults.

        """
        # Set current optimization problem
        self.pb = problem

        # Override configuration with provided keyword arguments
        for key, value in run_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Unknown config parameter: '{key}'")

        # Initialize population's arrays
        self.x = np.zeros((self.config.n_max, self.pb.n_var))
        self.dist = np.zeros((self.config.n_max, self.config.n_max))
        self.obj = np.zeros((self.config.n_max, self.pb.n_obj))
        self.const = np.zeros((self.config.n_max, self.pb.n_con))
        self.f = np.zeros(self.config.n_max)
        self.fs = np.zeros(self.config.n_max)
        self.mins = []

        # Initialize counters
        self.c_gen = 0
        self.c_pop = 0
        self.c_cross = 0
        self.c_mute = 0

        # Initialize results
        self.gen = []
        self.results = None

        # Initialize save directory
        if self.config.save:

            # Set save directory
            if self.config.save_dir is None:
                save_dir = Path.cwd() / "results" / self.pb.name
            else:
                save_dir = Path(self.config.save_dir)
                if save_dir.is_absolute():
                    save_dir = save_dir / self.pb.name
                else:
                    save_dir = Path.cwd() / save_dir / self.pb.name
            save_dir.mkdir(parents=True, exist_ok=True)
            self.config.save_dir = save_dir

            # Initialize results files
            files = ["gen.txt", "pop.txt", "mins.txt"]
            for file in files:
                with open(self.config.save_dir / file, "w") as f:
                    f.write("")

            # Initialize summary file
            config_dict = self.config.__dict__.copy()
            config_dict["save_dir"] = str(config_dict["save_dir"])
            pb_dict = {
                "name": self.pb.name,
                "vars": self.pb.vars,
                "bounds": self.pb.bounds,
                "objs": self.pb.objs,
                "consts": self.pb.consts,
                "limits": self.pb.limits,
                "sf": self.pb.sf,
            }
            self.summary = {"config": config_dict, "pb": pb_dict}
            with open(self.config.save_dir / "sum.yml", "w") as file:
                yaml.dump(self.summary, file, default_flow_style=False)

        # Print problem considered
        if self.config.display:
            print("################################################")
            print("#################### VIKGA #####################")
            print("################################################")
            self.pb.print()
            print("#################### RUN #######################")

        # Start timer
        self.st = time.time()

    def _initialize(self):
        """
        Generate and evaluate the initial population.
        """
        # Generate sample
        x_init = sample(self.config.n_min, self.pb.bounds, method=self.config.sample)

        # Evaluate initial population
        self._evaluate(x_init)

        # Initialize distance matrix
        self.dist[: self.c_pop, : self.c_pop] = (
            cdist(self.x[: self.c_pop], self.x[: self.c_pop], "euclidean") / self.pb.sf
        )

    def _evaluate(self, x: np.ndarray):
        """
        Evaluate new individuals and update the population.

        Parameters
        ----------
        x : np.ndarray
            Array of individuals to evaluate.
        """
        # Evaluate new individuals in parallel
        n_eval = len(x)
        if self.config.parallel:

            # Check number of proc
            n_proc = min(self.config.n_proc, n_eval)
            n_proc = max(2, n_proc)

            # Joblib wrapper for pickling
            @delayed
            @wrap_non_picklable_objects
            def func_wrapped(xi):
                return self.pb.func(xi)

            res = Parallel(n_jobs=n_proc)(func_wrapped(xi) for xi in x)

        # Evaluate new individuals sequentially
        else:
            res = [self.pb.func(xi) for xi in x]

        # Post process results
        res = np.array(res)
        x = np.round(x, self.config.decimals)
        obj = res[:, : self.pb.n_obj]
        const = res[:, self.pb.n_obj :]

        # Update population's arrays
        self.x[self.c_pop : self.c_pop + n_eval] = x
        self.obj[self.c_pop : self.c_pop + n_eval] = obj
        self.const[self.c_pop : self.c_pop + n_eval] = const
        self.c_pop += n_eval

        # Log new evaluations
        if self.config.save:
            self._log_pop(x, obj, const)

    def _evolve(self):
        """
        Check if the algorithm is evolving.

        Returns
        -------
        bool
            True if the algorithm has not converged, False otherwise.
        """
        # Check maximum population size
        if self.c_pop == self.config.n_max:
            if self.config.display:
                print("\nMaximum population size reached!")
            return False

        # Check maximum generation
        if self.c_gen == self.config.n_gen:
            if self.config.display:
                print("\nMaximum generation number reached!")
            return False

        # Check convergence
        if self.c_gen >= 2 * self.config.n_conv:
            recent_best = [g[-1] for g in self.gen[-self.config.n_conv :]]
            recent_xs = self.x[recent_best]
            recent_objs = self.obj[recent_best]
            obj_range = np.max(recent_objs) - np.min(recent_objs)
            std_per_var = np.std(recent_xs, axis=0)
            if (obj_range < 1e-6) and (np.all(std_per_var < 1e-6)):
                if self.config.display:
                    print(f"\nConvergence: obj_range={obj_range}, x_std={std_per_var} !")
                return False
        return True

    def _fitness(self):
        """
        Compute the fitness and scaled fitness of the current population.
        """
        # Consider current population values
        obj = self.obj[: self.c_pop, 0]
        const = self.const[: self.c_pop]
        dist = self.dist[: self.c_pop, : self.c_pop]

        # Compute fitness
        self.f[: self.c_pop], self.is_feasible, self.pen = compute_fitness(
            obj, const, alpha=self.config.alpha
        )

        # Compute scaled fitness
        self.fs[: self.c_pop], self.mins = scale_fitness(
            self.f[: self.c_pop], dist, self.is_feasible, d=self.config.d_mins, n=self.config.n_mins
        )

        # Print info in the console
        if self.config.display:
            print(
                f"\rGeneration {self.c_gen}: {self.c_pop} individuals and {len(self.mins)} local minima",
                end="",
            )

        # Log current generation results
        self.gen.append([self.c_pop, self.c_cross, self.c_mute, len(self.mins), self.mins[0]])
        if self.config.save:
            self._log_gen()

    def _crossover(self):
        """
        Perform crossover to generate new offspring.
        """

        # Init crossover offsprings
        self.x_cross = np.zeros((self.config.n_cross * 2, self.pb.n_var))

        # Get fitness-proportionate probability
        p_fit = self.fs[: self.c_pop] / np.sum(self.fs[: self.c_pop])

        # Loop on crossovers
        for i in range(self.config.n_cross):

            # Select first parents using fitness-proportionate probability and Debs ruled tournament
            n_select = 8
            selection = np.random.choice(self.c_pop, n_select, replace=False, p=p_fit)

            # Apply Deb's rules to pick the best first parent
            id_p1 = selection[0]
            for ind in selection[1:]:
                if all(~self.is_feasible[[ind, id_p1]]):
                    if self.pen[ind] < self.pen[id_p1]:
                        id_p1 = ind
                elif all(self.is_feasible[[ind, id_p1]]):
                    if self.fs[id_p1] < self.fs[ind]:
                        id_p1 = ind
                else:
                    id_p1 = np.array([ind, id_p1])[self.is_feasible[[ind, id_p1]]][0]

            # Select second parents using distance-proportionate probability related to the first parent
            id_pop = np.setdiff1d(np.arange(self.c_pop), id_p1)
            prox = 1.0 / self.dist[id_pop, id_p1]
            p_dist = prox / np.sum(prox)
            crowd = np.random.choice(id_pop, self.config.n_crowd, p=p_dist, replace=False)
            id_p2 = crowd[np.argmax(self.fs[crowd])]

            # Cross parents
            x_off1, x_off2 = cross(
                self.x[id_p1],
                self.x[id_p2],
                self.pb.bounds,
                method=self.config.crossover,
                eta=self.config.eta_c,
            )
            self.x_cross[i * 2] = x_off1
            self.x_cross[i * 2 + 1] = x_off2

    def _mutation(self):
        """
        Perform mutation to generate new offspring.
        """
        # Init mutant offsprings
        self.x_mute = np.zeros((self.config.n_mute, self.pb.n_var))

        # Select random individuals to mutate from
        i_mute = np.random.choice(self.c_pop, size=self.config.n_mute, replace=False)

        # Loop on mutations
        for k in range(self.config.n_mute):

            # Mute individual
            self.x_mute[k] = mutate(
                self.x[i_mute[k]],
                self.pb.bounds,
                method=self.config.mutation,
                eta=self.config.eta_m,
            )

    def _addition(self):
        """
        Add new offsprings to the population.
        """

        # Loop on all offsprings
        x_off = np.vstack((self.x_cross, self.x_mute))
        id_add = []
        c_add = [0, 0]
        for i, x in enumerate(x_off):
            if self.c_pop + sum(c_add) < self.config.n_max:

                # Get closest individual
                dists = np.linalg.norm(self.x[: self.c_pop] - x, axis=1) / self.pb.sf
                id_nei = np.argmin(dists)

                # Check distance threshold to add offspring
                if dists[id_nei] > self.config.d0 * (
                    1.001 - self.fs[id_nei] * (1 - 0.5 * (0.9**self.c_gen))
                ):
                    id_add.append(i)

                    # Count crossovers/mutations
                    if i < len(self.x_cross):
                        c_add[0] += 1
                    else:
                        c_add[1] += 1

        # Update counters
        self.c_cross = c_add[0]
        self.c_mute = c_add[1]

        # Evaluate new individuals
        if sum(c_add) > 0:
            self._evaluate(x_off[id_add])

            # update distance matrix
            x_add = self.x[self.c_pop - sum(c_add) : self.c_pop]
            dist0 = cdist(x_add, self.x[: self.c_pop - sum(c_add)], "euclidean") / self.pb.sf
            dist1 = cdist(x_add, x_add, "euclidean") / self.pb.sf
            self.dist[: self.c_pop, : self.c_pop] = np.block(
                [
                    [self.dist[: self.c_pop - sum(c_add), : self.c_pop - sum(c_add)], dist0.T],
                    [dist0, dist1],
                ]
            )

    def _stop(self):
        # Stop timer
        self.et = time.time()

        # Get results
        x = self.x[: self.c_pop]
        obj = self.obj[: self.c_pop]
        const = self.const[: self.c_pop]
        f = self.f[: self.c_pop, np.newaxis]
        fs = self.fs[: self.c_pop, np.newaxis]
        pop = np.concatenate([x, obj, const, f, fs], axis=1)
        gen = np.array(self.gen)

        # Update and save summary
        self.summary["res"] = {
            "run_time": round(self.et - self.st, 2),
            "n_eval": self.c_pop,
            "mins": self.mins,
            "x": self.x[self.mins[0]].tolist(),
            "obj": self.obj[self.mins[0]].tolist(),
            "const": self.const[self.mins[0]].tolist(),
        }
        if self.config.save:
            with open(self.config.save_dir / "sum.yml", "w") as file:
                yaml.dump(self.summary, file, default_flow_style=False)

        # Make results object
        self.results = Results(self.summary, pop, gen)

        # Print and plot results
        if self.config.display:
            print()
            self.results.print()
        if self.config.plot:
            self.results.plot()

    def _log_pop(self, x: np.ndarray, obj: np.ndarray, const: np.ndarray):
        """
        Log the population data.

        Parameters
        ----------
        x : np.ndarray
            Array of individuals to log.
        obj : np.ndarray
            Array of objectives to log.
        const : np.ndarray
            Array of constraints to log.
        """
        # Prepare lines to write
        lines = []
        for i in range(len(x)):
            data = [str(v) for v in x[i]] + [str(v) for v in obj[i]] + [str(v) for v in const[i]]
            line = " ".join(data) + "\n"
            lines.append(line)

        # Write all lines at once
        fpath = os.path.join(self.config.save_dir, "pop.txt")
        with open(fpath, "a") as file:
            file.writelines(lines)

    def _log_gen(self):
        """
        Log the generation data.
        """
        # Log generation data
        line = f"{self.c_pop} {self.c_cross} {self.c_mute} {len(self.mins)} {self.mins[0]}\n"
        fpath = os.path.join(self.config.save_dir, "gen.txt")
        with open(fpath, "a") as file:
            file.write(line)
