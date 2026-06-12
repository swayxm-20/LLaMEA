import ioh
import numpy as np
import importlib.util
import sys
import os

class IOHRunner:
    """
    Handles the evaluation of algorithms using IOHexperimenter (IOH) suite.
    """

    def __init__(self, budget, dims):
        """
        Initializes the IOHRunner.

        Args:
            budget (int): The evaluation budget for each algorithm run.
            dims (int): The dimensionality of the optimization problems.
        """
        self.budget = budget
        self.dims = dims
        # BBOB suite with 24 problems, 3 instances each
        self.suite = ioh.suite.BBOB(problem_ids=list(range(1, 25)),
                                     instances=list(range(1, 4)),
                                     dimensions=[self.dims])

    def _calculate_aocc(self, log_info):
        """
        Flatten deeply nested IOH logger data and extract numeric f-values.

        Args:
            log_info: Nested dictionary or list from IOH logger.

        Returns:
            float: The computed AOCC value.
        """
        if not log_info:
            return 0.0

        data_points = []

        # Candidate keys for best-so-far fitness
        key_candidates = [
            "current_y_best", "CURRENTBESTY", "CurrentBestY",
            "current_y", "CURRENTY", "CurrentY",
            "rawy", "RAWY", "rawybest", "RAWYBEST",
            "best_so_far_fvalue"
        ]

        def recurse_extract(entry):
            """Recursively extract numeric values from dicts/lists."""
            if isinstance(entry, dict):
                for k, v in entry.items():
                    if k in key_candidates and isinstance(v, (float, int)) and np.isfinite(v):
                        data_points.append(v)
                    else:
                        recurse_extract(v)
            elif isinstance(entry, list):
                for item in entry:
                    recurse_extract(item)
            elif isinstance(entry, (float, int)):
                data_points.append(entry)

        recurse_extract(log_info)

        if not data_points:
            print(f"Warning: No valid f-values found in log_info (type={type(log_info)})")
            return 0.0

        # Compute AOCC
        data = np.array(data_points[:self.budget])
        lb, ub = 1e-8, 1e2
        clipped = np.clip(data, lb, ub)
        normalized = 1 - (clipped - lb) / (ub - lb)

        if len(normalized) < self.budget:
            padding = np.full(self.budget - len(normalized), normalized[-1])
            normalized = np.concatenate([normalized, padding])

        return float(np.mean(normalized))

    def evaluate(self, algorithm_path):
        """
        Evaluates a given algorithm file on the IOH suite.

        Args:
            algorithm_path (str): The path to the Python file of the algorithm.

        Returns:
            float: The average AOCC score over all problems in the suite.
        """
        # Dynamically import the algorithm
        spec = importlib.util.spec_from_file_location("generated_algorithm", algorithm_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_algorithm"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, 'CustomAlgorithm'):
            raise AttributeError("The algorithm file must contain a class named 'CustomAlgorithm'")
        algorithm_class = module.CustomAlgorithm

        algo_name = os.path.basename(algorithm_path).replace('.py', '')
        folder_name = f"{algo_name}_{np.random.randint(100000)}"
        analyzer_logger = ioh.logger.Analyzer(root="ioh_data", folder_name=folder_name)

        all_runs_aocc = []

        # Iterate over problems using the iterator (ioh.iohcpp suite is not subscriptable)
        for problem in self.suite:
            # Ensure bounds are numpy arrays
            lb_raw = np.array(problem.bounds.lb)
            ub_raw = np.array(problem.bounds.ub)
            lb_array = np.full(self.dims, lb_raw) if lb_raw.ndim == 0 else lb_raw
            ub_array = np.full(self.dims, ub_raw) if ub_raw.ndim == 0 else ub_raw

            # Run 3 independent trials
            for _ in range(3):
                # Setup logger properly
                store_logger = ioh.logger.Store(
                    triggers=[ioh.iohcpp.logger.trigger.ALWAYS],
                    properties=[ioh.iohcpp.logger.property.CURRENTBESTY]
                )
                logger_combined = ioh.logger.Combine([analyzer_logger, store_logger])
                problem.attach_logger(logger_combined)

                # Instantiate algorithm
                algo_instance = algorithm_class(
                    max_evaluations=self.budget,
                    dims=self.dims,
                    lb=lb_array,
                    ub=ub_array
                )

                # Run the algorithm
                algo_instance(problem)

                # Detach logger
                problem.detach_logger()

                # Extract logged data
                logged_data = store_logger.data()

                run_aocc = self._calculate_aocc(logged_data)
                all_runs_aocc.append(run_aocc)

                problem.reset()

        analyzer_logger.close()

        return np.mean(all_runs_aocc) if all_runs_aocc else 0.0
