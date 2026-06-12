import numpy as np

class CustomAlgorithm:
    """
    Adaptive Memetic Differential Evolution with OppositionBased Learning (AMDEOBL).

    Main ideas
    ----------
    * Standard JADE/DE/currenttopbest/1 mutation with adaptive F and CR.
    * A small historic memory (size H) stores successful F and CR values and is
      updated with Lehmer means (SHADEstyle).
    * OppositionBased Initialization: the opposite points of the initial
      population are evaluated and the best `pop_size` individuals are kept.
    * Periodic Gaussian perturbation for finegrained local search.
    * When stagnation is detected, the worst half of the population is
      reinitialized by the opposite of the current best (plus a small random
      offset)  a cheap diversification operator.
    * All candidates are clipped to the provided bounds.
    * Works with the IOHexperimenter ``Problem`` interface:
          - ``problem(x)`` returns a fitness value.
          - ``problem.state.evaluations`` holds the current evaluation count.
          - ``problem.state.optimum_found`` signals early stopping.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(self, max_evaluations, dims, lb, ub):
        """
        Parameters
        ----------
        max_evaluations : int
            Total evaluation budget.
        dims : int
            Dimensionality of the problem.
        lb, ub : np.ndarray, shape (dims,)
            Lower and upper bounds of the search space.
        """
        self.max_evaluations = int(max_evaluations)
        self.dims = int(dims)
        self.lb = lb.astype(float)
        self.ub = ub.astype(float)

        # best solution seen so far
        self.best_x = None
        self.best_f = np.inf

        # population size (at least 8, scaled with dimensionality)
        self.pop_size = max(8, int(0.3 * self.dims))

        # JADE / SHADE memory size
        self.H = 5
        self.MF = np.full(self.H, 0.5)   # historic scaling factors
        self.MCR = np.full(self.H, 0.9)  # historic crossover probs
        self.mem_idx = 0                 # cyclic index for memory update

        # pbest selection percentage
        self.p_best = 0.2

        # intervals for extra operators
        self.gauss_interval = max(20, int(0.015 * self.max_evaluations))
        self.stagnation_limit = max(200, int(0.05 * self.max_evaluations))

        # bookkeeping
        self._last_gauss = 0
        self._last_improve = 0

        # random number generator (can be seeded externally)
        self.rng = np.random.default_rng()

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _ensure_bounds(self, x):
        """Clip a vector to the searchspace bounds."""
        return np.clip(x, self.lb, self.ub)

    def _initialize_population(self):
        """Uniform random initialisation inside the bounds."""
        return self.rng.uniform(self.lb, self.ub,
                                size=(self.pop_size, self.dims))

    def _opposite(self, pop):
        """Opposition of a set of points w.r.t. the box bounds."""
        return self.lb + self.ub - pop

    def _sample_F_CR(self):
        """Draw adaptive F and CR from the historic memories."""
        idx = self.rng.integers(self.H)

        # Cauchy for F (JADE style)
        Fi = self.rng.standard_cauchy() * 0.1 + self.MF[idx]
        while Fi <= 0:
            Fi = self.rng.standard_cauchy() * 0.1 + self.MF[idx]
        Fi = min(Fi, 1.0)

        # Normal for CR (SHADE style)
        CRi = self.rng.normal(self.MCR[idx], 0.1)
        CRi = np.clip(CRi, 0.0, 1.0)

        return Fi, CRi, idx

    def _mutate(self, pop, i, Fi, pbest):
        """JADE currenttopbest/1 mutation."""
        idxs = list(range(self.pop_size))
        idxs.remove(i)
        r1, r2 = self.rng.choice(idxs, size=2, replace=False)

        xi = pop[i]
        xr1 = pop[r1]
        xr2 = pop[r2]

        mutant = xi + Fi * (pbest - xi) + Fi * (xr1 - xr2)
        return self._ensure_bounds(mutant)

    def _crossover(self, target, mutant, CRi):
        """Binomial crossover (ensuring at least one mutant component)."""
        mask = self.rng.random(self.dims) < CRi
        if not np.any(mask):
            mask[self.rng.integers(self.dims)] = True
        trial = np.where(mask, mutant, target)
        return self._ensure_bounds(trial)

    def _gaussian_perturb(self, x, sigma_factor=0.02):
        """Small Gaussian perturbation for local search."""
        sigma = sigma_factor * (self.ub - self.lb)
        pert = self.rng.normal(0.0, sigma, size=self.dims)
        return self._ensure_bounds(x + pert)

    # --------------------------------------------------------------------- #
    # Main optimisation loop
    # --------------------------------------------------------------------- #
    def __call__(self, problem):
        # ---------------------- initial population ---------------------- #
        pop = self._initialize_population()
        opp = self._opposite(pop)
        # evaluate both sets
        combined = np.vstack((pop, opp))
        fitness = np.empty(combined.shape[0])

        for i in range(combined.shape[0]):
            f = problem(combined[i])
            if not np.isfinite(f):
                f = 1e8
            fitness[i] = f
            if f < self.best_f:
                self.best_f = f
                self.best_x = combined[i].copy()
                self._last_improve = problem.state.evaluations

        # keep the best `pop_size` individuals
        best_idx = np.argsort(fitness)[:self.pop_size]
        pop = combined[best_idx]
        fitness = fitness[best_idx]

        # ---------------------- evolutionary process -------------------- #
        while problem.state.evaluations < self.max_evaluations:
            if getattr(problem.state, "optimum_found", False):
                break

            # ----- pbest selection (random among top p% ) ----- #
            n_pbest = max(1, int(self.p_best * self.pop_size))
            top_indices = np.argsort(fitness)[:n_pbest]
            pbest_choices = self.rng.choice(top_indices,
                                            size=self.pop_size,
                                            replace=True)
            pbest_pop = pop[pbest_choices]

            # ----- containers for successful parameters ----- #
            succ_F, succ_CR, succ_delta = [], [], []

            # ----- DE generation ----- #
            for i in range(self.pop_size):
                Fi, CRi, mem_id = self._sample_F_CR()
                pbest = pbest_pop[i]

                mutant = self._mutate(pop, i, Fi, pbest)
                trial = self._crossover(pop[i], mutant, CRi)

                f_trial = problem(trial)
                if not np.isfinite(f_trial):
                    f_trial = 1e8

                if f_trial < fitness[i]:
                    delta = fitness[i] - f_trial
                    succ_F.append(Fi)
                    succ_CR.append(CRi)
                    succ_delta.append(delta)

                    pop[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.best_f:
                        self.best_f = f_trial
                        self.best_x = trial.copy()
                        self._last_improve = problem.state.evaluations

                # early exit checks
                if getattr(problem.state, "optimum_found", False):
                    break
                if problem.state.evaluations >= self.max_evaluations:
                    break

            # ----- memory update (Lehmer mean for F, arithmetic for CR) ----- #
            if succ_F:
                w = np.array(succ_delta) / np.sum(succ_delta)
                MF_new = np.sum(w * np.square(succ_F)) / np.sum(w * succ_F)
                MCR_new = np.sum(w * succ_CR)

                self.MF[self.mem_idx] = MF_new
                self.MCR[self.mem_idx] = MCR_new
                self.mem_idx = (self.mem_idx + 1) % self.H

            # ----- Gaussian perturbation (every gauss_interval) ----- #
            if (problem.state.evaluations - self._last_gauss) >= self.gauss_interval:
                n_gauss = max(1, int(0.1 * self.pop_size))
                chosen = self.rng.choice(self.pop_size, size=n_gauss, replace=False)
                for idx in chosen:
                    cand = self._gaussian_perturb(pop[idx])
                    f_cand = problem(cand)
                    if not np.isfinite(f_cand):
                        f_cand = 1e8
                    if f_cand < fitness[idx]:
                        pop[idx] = cand
                        fitness[idx] = f_cand
                        if f_cand < self.best_f:
                            self.best_f = f_cand
                            self.best_x = cand.copy()
                            self._last_improve = problem.state.evaluations
                self._last_gauss = problem.state.evaluations

            # ----- Stagnation detection & oppositionbased restart ----- #
            if (problem.state.evaluations - self._last_improve) >= self.stagnation_limit:
                n_restart = self.pop_size // 2
                worst_idx = np.argsort(fitness)[-n_restart:]

                # opposite of the best (plus tiny random noise)
                opp_best = self._ensure_bounds(self.lb + self.ub - self.best_x)
                noise = self.rng.uniform(-0.01, 0.01, size=(n_restart, self.dims))
                new_cands = self._ensure_bounds(opp_best + noise)

                for idx, cand in zip(worst_idx, new_cands):
                    f_cand = problem(cand)
                    if not np.isfinite(f_cand):
                        f_cand = 1e8
                    pop[idx] = cand
                    fitness[idx] = f_cand
                    if f_cand < self.best_f:
                        self.best_f = f_cand
                        self.best_x = cand.copy()
                        self._last_improve = problem.state.evaluations

                # reset stagnation counter
                self._last_improve = problem.state.evaluations

        return self.best_x