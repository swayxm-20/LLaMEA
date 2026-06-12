import numpy as np

class CustomAlgorithm:
    """
    A robust Random Search optimizer with logging-safe behavior.

    This version ensures that:
    1. Each evaluation is finite (NaN/inf replaced with a large number).
    2. The best-so-far solution (`best_x`) and fitness (`best_f`) are tracked.
    3. Always calls `problem(x)` to generate valid log entries.
    4. Compatible with IOHexperimenter's Store logger and AOCC calculation.
    """
    def __init__(self, max_evaluations, dims, lb, ub):
        """
        Args:
            max_evaluations (int): evaluation budget
            dims (int): problem dimensionality
            lb (np.ndarray): lower bounds (numpy array)
            ub (np.ndarray): upper bounds (numpy array)
        """
        self.max_evaluations = max_evaluations
        self.dims = dims
        self.lb = lb
        self.ub = ub

        # Track the best solution
        self.best_x = None
        self.best_f = np.inf

    def _ensure_bounds(self, x):
        """Clip a candidate solution to the problem bounds."""
        return np.clip(x, self.lb, self.ub)

    def __call__(self, problem):
        """
        Main optimization loop.
        """
        # Initialize best solution with a random point
        x0 = np.random.uniform(self.lb, self.ub, self.dims)
        f0 = problem(x0)
        if not np.isfinite(f0):
            f0 = 1e8
        self.best_x = x0.copy()
        self.best_f = f0

        # Evaluate random solutions until budget exhausted
        while problem.state.evaluations < self.max_evaluations:
            if problem.state.optimum_found:
                break

            x = np.random.uniform(self.lb, self.ub, self.dims)
            x = self._ensure_bounds(x)

            f_val = problem(x)
            if not np.isfinite(f_val):
                f_val = 1e8

            # Update best-so-far
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()

        return self.best_x
