import time
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed, wrap_non_picklable_objects
from .config import Config
from .problem import Problem
from .logger import Logger
from .operators.samples import sample
from .fitness import compute_fitness, scale_fitness, get_optima
from .operators.crossovers import cross
from .operators.mutations import mutate


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
        Initialize the VIKGA object.
        """
        # Initialize default configuration
        self.config = Config()

        # Initialize problem to solve
        self.pb = None

        # Initialize population's arrays
        self.x = None
        self.x_cross = None
        self.x_mute = None
        self.dist = None
        self.obj = None
        self.const = None
        self.pen = None
        self.f = None
        self.fs = None
        self.mins = []

        # Initialize counters
        self.c_gen = 0
        self.c_pop = 0
        self.c_cross = 0
        self.c_mute = 0
        self.c_reject = 0

        # Initialize logger
        self.logger = None

    def run(self, pb: Problem, **run_kwargs):
        """
        Run the algorithm for a given optimization problem.

        Parameters
        ----------
        pb : Problem
            The optimization problem to solve.
        **run_kwargs : dict, optional
            Keyword arguments to override the algorithm's configuration parameters.
            See the "Config" class for available parameters.
        """
        # Start optimization
        self._start(pb, **run_kwargs)

        # Initialize population
        self._initialize()

        # Loop over generations
        for self.c_gen in range(1, self.config.n_gen + 1):

            # Perform crossover, mutation and addition operation
            self._crossover()
            self._mutation()
            self._addition()

            # Compute current population fitness
            self._fitness()

            # Check convergence
            if self._convergence():
                break

        # Stop optimization process
        self._stop()

    def run_multiple(self, pb: Problem, n_run=2, **run_kwargs):
        """
        Run the algorithm multiple times for a given optimization problem.

        Parameters
        ----------
        pb : Problem
            The optimization problem to solve.
        n_run : int, optional
            Number of runs, by default 2.
        **run_kwargs : dict, optional
            Keyword arguments to override the algorithm's configuration parameters.
            See the "Config" class for available parameters.
        """

        # Loop on runs
        for k in range(n_run):
            self.run(pb, **run_kwargs, save_dir=f"run_{k}")

    def _start(self, pb: Problem, **run_kwargs):
        """
        Start the optimization process.

        Parameters
        ----------
        pb : Problem
            The optimization problem to solve.
        **run_kwargs : dict, optional
            Keyword arguments to override the algorithm's configuration parameters.
            See the "Config" class for available parameters.

        """
        # Set optimization problem
        self.pb = pb

        # Override configuration with provided keyword arguments
        for key, value in run_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                warnings.warn(f"Unknown config parameter: '{key}'", UserWarning)

        # Set logger
        if self.config.save:
            self.logger = Logger(self.pb, self.config)

        # Set population's arrays
        self.x = np.zeros((self.config.n_max, self.pb.n_var))
        self.x_cross = np.zeros((self.config.n_cross, self.pb.n_var))
        self.x_mute = np.zeros((self.config.n_mute, self.pb.n_var))
        self.dist = np.zeros((self.config.n_max, self.config.n_max))
        self.obj = np.zeros((self.config.n_max, self.pb.n_obj))
        self.const = np.zeros((self.config.n_max, self.pb.n_con))
        self.pen = np.zeros(self.config.n_max)
        self.f = np.zeros(self.config.n_max)
        self.fs = np.zeros(self.config.n_max)
        self.mins = []

        # Set counters
        self.c_gen = 0
        self.c_pop = 0
        self.c_cross = 0
        self.c_mute = 0
        self.c_reject = 0

        # Display considered problem
        if self.config.display:
            print("################################################")
            print("#################### VIKGA #####################")
            print("################################################")
            print("################### PROBLEM ####################")
            print(f"Name:   {self.pb.name}")
            print(f"N° vars:   {self.pb.n_var}")
            print(f"N° objs:   {self.pb.n_obj}")
            print(f"N° consts: {self.pb.n_con}")
            print("##################### RUN ######################")

        # Start timer
        self.st = time.time()

    def _initialize(self):
        """
        Initialize the population.
        """
        # Sample design variables within boundaries
        x_init = sample(self.config.n_min, self.pb.bounds, method=self.config.sample)
        x_init = np.round(x_init, self.config.decimals)

        # Evaluate initial population
        obj, const = self._evaluate(x_init)

        # Update algorithm's population
        self._update(x_init, obj, const)

        # Log initial population
        if self.config.save:
            self.logger.log_pop(x_init, obj, const)

        # Compute initial population fitness
        self._fitness()

    def _evaluate(self, x: np.ndarray):
        """
        Evaluate new individuals to get objectives and constraints values.

        Parameters
        ----------
        x : np.ndarray
            Design variables to evaluate.

        Returns
        -------
        obj : np.ndarray
            Objective values.
        const : np.ndarray
            Constraints values.
        """
        # Get number of evaluations
        n_eval = len(x)

        # Evaluate in parallel
        if self.config.parallel:

            # If more than one evaluations
            if n_eval > 1:

                # Check number of process to use
                n_proc = min(self.config.n_proc, n_eval)

                # Joblib wrapper for pickling issues
                @delayed
                @wrap_non_picklable_objects
                def func_wrapped(xi):
                    return self.pb.func(xi)

                # Evaluate individuals in parallel
                res = Parallel(n_jobs=n_proc)(func_wrapped(xi) for xi in x)

            else:
                # Evaluate individual
                res = [self.pb.func(x[0])]

        # Evaluate individuals sequentially
        else:
            res = [self.pb.func(xi) for xi in x]

        # Post process results
        res = np.array(res)
        obj = res[:, : self.pb.n_obj]
        const = res[:, self.pb.n_obj :]

        return obj, const

    def _update(self, x, obj, const):
        """
        Update algorithm's population arrays.

        Parameters
        ----------
        x : np.ndarray
            Design variables.
        obj : np.ndarray
            Objective values.
        const : np.ndarray
            Constraints values.
        """
        # Get number of new individuals
        n = len(x)

        # Update population's arrays
        self.x[self.c_pop : self.c_pop + n] = x
        self.obj[self.c_pop : self.c_pop + n] = obj
        self.const[self.c_pop : self.c_pop + n] = const

        # Update distance matrix
        dist0 = self.dist[: self.c_pop, : self.c_pop]
        dist1 = cdist(x, self.x[: self.c_pop], "euclidean") / self.pb.sf
        dist2 = cdist(x, x, "euclidean") / self.pb.sf
        self.dist[: self.c_pop + n, : self.c_pop + n] = np.block(
            [[dist0, dist1.T], [dist1, dist2]]
        )

        # Update population size
        self.c_pop += n

    def _fitness(self):
        """
        Compute fitness and scaled fitness of the current population.
        """
        # Consider current population values
        obj = self.obj[: self.c_pop]
        const = self.const[: self.c_pop]

        # Compute current population fitness
        self.f[: self.c_pop], self.pen[: self.c_pop] = compute_fitness(obj, const)

        # Compute current population scaled fitness
        self.fs[: self.c_pop], self.mins = scale_fitness(
            self.f[: self.c_pop],
            self.pen[: self.c_pop],
            self.dist[: self.c_pop, : self.c_pop],
            d=self.config.d_mins,
            n=self.config.n_mins,
        )

        # Display current generation info
        if self.config.display:
            print(
                f"\rGeneration {self.c_gen}: {self.c_pop} individuals and {len(self.mins)} local minima",
                end="",
            )

        # Log current generation results
        if self.config.save:
            data = [
                self.c_pop,
                self.c_cross,
                self.c_mute,
                self.c_reject,
                len(self.mins),
                self.mins[0],
            ]
            self.logger.log_gen(data)

    def _crossover(self):
        """
        Perform crossover operation.
        """
        # Compute fitness-proportionate probability
        p_fit = self.fs[: self.c_pop] / np.sum(self.fs[: self.c_pop])

        # Loop on crossovers
        for i in range(self.config.n_cross):

            # Select first parents using fitness-proportionate probability
            id_p1 = np.random.choice(len(p_fit), p=p_fit)

            # Select second parents using first parent distance-proportionate probability
            prox = 1.0 / (1e-12 + self.dist[: self.c_pop, id_p1])
            p_dist = prox / np.sum(prox)
            crowd = np.random.choice(len(p_dist), 8, p=p_dist, replace=False)
            id_p2 = crowd[np.argmax(self.fs[crowd])]

            # Cross parents
            self.x_cross[i] = cross(
                self.x[id_p1],
                self.x[id_p2],
                self.pb.bounds,
                method=self.config.crossover,
                eta=self.config.eta_c,
            )

    def _mutation(self):
        """
        Perform mutation operation.
        """
        # Loop on mutations
        for k in range(self.config.n_mute):

            # Select random individuals to mutate
            id_mute = np.random.choice(self.c_pop)

            # Mute individual
            self.x_mute[k] = mutate(
                self.x[id_mute],
                self.pb.bounds,
                method=self.config.mutation,
                eta=self.config.eta_m,
                pm=self.config.pm,
            )

    def _addition(self):
        """
        Add new offsprings to the population.
        """
        # Reset counters
        id_add = []
        id_reject = []
        self.c_cross = 0
        self.c_mute = 0

        # Loop on all offsprings
        x_off = np.vstack((self.x_cross, self.x_mute))
        x_off = np.round(x_off, self.config.decimals)
        for i, x in enumerate(x_off):
            if self.c_pop + len(id_add) < self.config.n_max:

                # Get closest individual
                dists = cdist([x], self.x[: self.c_pop])[0] / self.pb.sf
                idx = np.argmin(dists)

                # Check distance threshold to add offspring
                if dists[idx] > 0.08 * (
                    1.001 - self.fs[idx] * (1 - 0.5 * (0.9 ** (self.c_gen + 1)))
                ):
                    # if dists[idx] > 0.1 * (1 - self.fs[idx] ** 2 * 0.95):

                    # Add individuals and update counters
                    id_add.append(i)
                    if i < len(self.x_cross):
                        self.c_cross += 1
                    else:
                        self.c_mute += 1

                # Reject individual
                else:
                    id_reject.append(i)
                    self.c_reject += 1

            # Maximum population size reached
            else:
                break

        # Evaluate new individuals
        if len(id_add) > 0:
            obj, const = self._evaluate(x_off[id_add])

            # Update algorithm's population
            self._update(x_off[id_add], obj, const)

            # Log new individuals
            if self.config.save:
                self.logger.log_pop(x_off[id_add], obj, const)

        # Log rejected population
        if self.config.save:
            self.logger.log_reject(x_off[id_reject])

    def _convergence(self):
        """
        Check if the algorithm converged.

        Returns
        -------
        bool
            True if the algorithm converged, False otherwise.
        """
        # Check maximum population size
        if self.c_pop == self.config.n_max:
            if self.config.display:
                print("\nMaximum population size reached!")
            return True

        # Check maximum number of generation
        if self.c_gen == self.config.n_gen:
            if self.config.display:
                print("\nMaximum generation number reached!")
            return True

        # Check distances from each local optimum to its nearest neighbor
        mins_dist = np.sort(self.dist[self.mins, : self.c_pop])[:, 1]
        if sum(mins_dist) < 1e-4:
            if self.config.display:
                print(f"\nConvergence: sum={sum(mins_dist)}, mean={np.mean(mins_dist)}")
            return True

        return False

    def _stop(self):
        """
        Stop the optimization process.
        """
        # Stop timer
        self.et = time.time()

        # Display results
        if self.config.display:
            print("################### RESULTS ####################")
            print(f"N° gens:  {self.c_gen}")
            print(f"N° evals: {self.c_pop}")
            print(f"N° saves: {self.c_reject}")
            print(f"Run time: {round(self.et - self.st, 2)} s")
            print("Local minima:")
            for x, obj in zip(self.x[self.mins], self.obj[self.mins, 0]):
                print(f"x={x} | obj={obj}")
            print("################################################")
