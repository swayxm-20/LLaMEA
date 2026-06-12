import re
import os
import csv
import traceback
from datetime import datetime
from ioh_runner import IOHRunner
from groq_caller import GroqCaller
from gemini_caller import GeminiCaller

class LLaMEA:
    """
    Implements the Large Language Model Evolutionary Algorithm (LLaMEA) framework.

    This class orchestrates the evolutionary process of generating and refining
    metaheuristic algorithms using a Large Language Model (LLM). It handles
    the initialization, evaluation, and evolution of algorithms based on their
    performance on the IOHexperimenter benchmark suite.
    """
    def __init__(self, api_key=None, budget=1000, dims=5, generations=20, mu=5, lambda_=5, selection_strategy='plus', llm_model='gemini-2.5-pro'):
        """
        Initializes the LLaMEA framework.

        Args:
            gemini_api_key (str): The API key for the Gemini API.
            budget (int): The evaluation budget for each algorithm run.
            dims (int): The dimensionality of the optimization problems.
            generations (int): The number of generations for the evolutionary process.
            mu (int): The parent population size.
            lambda_ (int): The offspring population size (using lambda_ to avoid keyword conflict).
            selection_strategy (str): 'plus' for (mu + lambda) or 'comma' for (mu, lambda).
            llm_model (str): The specific Gemini model to use.
        """
        self.gemini_caller = GeminiCaller(api_key=api_key, model=llm_model)
        self.ioh_runner = IOHRunner(budget, dims)
        self.generations = generations
        self.mu = mu
        self.lambda_ = lambda_
        self.selection_strategy = selection_strategy.lower()
        if self.selection_strategy not in ['plus', 'comma']:
            raise ValueError("selection_strategy must be 'plus' or 'comma'")
        if self.selection_strategy == 'comma' and self.lambda_ < self.mu:
            print("Warning: (mu, lambda) strategy requires lambda >= mu. Forcing lambda = mu.")
            self.lambda_ = self.mu

        self.population = [] # Will store the top 'mu' individuals
        self.history = [] # Will store all evaluated individuals
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"log_mu{self.mu}_lambda{self.lambda_}_{self.selection_strategy}_{timestamp}.csv"
        
        # Write the header to the new CSV file
        try:
            with open(self.log_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['generation', 'best_aocc', 'best_algo_name'])
            print(f"Logging convergence data to {self.log_filename}")
        except IOError as e:
            print(f"Warning: Could not create log file {self.log_filename}. Error: {e}")

    def _get_initial_prompt(self):
        with open("random_search.py", "r") as f:
            random_search_code = f.read()

        return f"""
Your task is to design novel metaheuristic algorithms to solve closed box optimization problems.
The optimization algorithm should handle a wide range of tasks, which is evaluated on a large test suite of noiseless functions.
Your task is to write the optimization algorithm in Python code.

The code MUST contain one class named 'CustomAlgorithm' with the following structure:
1. An `__init__` method with the exact signature: `__init__(self, max_evaluations, dims, lb, ub)`
   - `lb` and `ub` are the lower and upper bounds. You can assume they are ALWAYS numpy arrays of shape (dims,).
2. A `__call__` method with the signature: `__call__(self, problem)`
   - This method will contain the main optimization loop and call `problem(x)` to evaluate solutions.

IMPORTANT: The only external library you can import is `numpy`. Do not use any other libraries like `scipy`, `pandas`, etc.

An example of a valid algorithm is as follows:
{random_search_code}

Give a novel heuristic algorithm to solve this task.
Give the response in the format:
# Name: <name of the algorithm>
# Code:
```python
<code>
```
"""

    def _extract_code(self, response):
        """
        Extracts the algorithm name and code from the LLM's response.

        Args:
            response (str): The response from the LLM.

        Returns:
            tuple: (algorithm_name, extracted_code)
                or (None, None) if extraction fails.
        """
        name_match = re.search(r"#\s*Name:\s*(.+)", response)
        if name_match:
            name = name_match.group(1).strip()
        else:
            class_match = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:\(]", response)
            if class_match:
                name = class_match.group(1).strip()
            else:
                name = "UnnamedAlgorithm"
        
        code_match = re.search(r"```python\n(.*)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None

        return name, code

    def _save_and_evaluate(self, name, code):
        """
        Saves the generated algorithm to a file and evaluates its performance.

        Args:
            name (str): The name of the algorithm.
            code (str): The Python code of the algorithm.

        Returns:
            tuple: A tuple containing the AOCC score (float) and an error message (str or None).
        """
        if not os.path.exists("generated_algos"):
            os.makedirs("generated_algos")
        
        # Sanitize name for filepath
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
        filepath = f"generated_algos/{safe_name}.py"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            score = self.ioh_runner.evaluate(filepath)
            return score, None
        except Exception as e:
            # --- Enhanced Error Logging ---
            # Capture the full traceback to get the exact line number and context of the error.
            error_message = traceback.format_exc()
            print(f"Error evaluating {name}:\n{error_message}")
            return 0.0, error_message

    def _build_offspring_prompt(self, parent):
        """
        Builds a prompt to generate an offspring, using a selected parent as context.

        Args:
            parent (dict): The parent individual dictionary.

        Returns:
            str: The prompt for the LLM.
        """
        prompt_parts = [self._get_initial_prompt()]

        # Add history summary
        history_summary_lines = []
        for h in self.history:
            error_str = f", Error: {h['error']}" if h['error'] else ""
            history_summary_lines.append(f"# {h['name']}: Score={h['score']:.4f}{error_str}")
        prompt_parts.append("\nPreviously generated algorithms and their results:")
        prompt_parts.append("\n".join(history_summary_lines))

        # Add parent context
        prompt_parts.append(f"\nYou are generating an offspring. Please use the following parent algorithm as inspiration.")
        prompt_parts.append(f"Parent '{parent['name']}' (Score: {parent['score']:.4f}):")
        if parent['error']:
            prompt_parts.append(f"Note: This parent failed with an error: {parent['error']}")
        prompt_parts.append(f"```python\n{parent['code']}\n```")
        prompt_parts.append("\nPlease generate a new, mutated, or improved version of this algorithm, or design a completely new one.")
        
        return "\n".join(prompt_parts)

    def _select_next_generation(self, offspring_population):
        """
        Selects the top 'mu' individuals for the next generation.

        Args:
            offspring_population (list): A list of newly generated offspring.

        Returns:
            list: The new population (top 'mu' individuals).
        """
        candidates = []
        if self.selection_strategy == 'plus':
            candidates = self.population + offspring_population
        else:  # 'comma' strategy
            candidates = offspring_population

        # Sort candidates: high score first, errors last
        def sort_key(ind):
            if ind['error']:
                return -float('inf')  # Errors are always worst
            return ind['score']

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        
        # Return the top 'mu'
        return sorted_candidates[:self.mu]

    def _log_convergence(self, generation, best_aocc, best_name):
        """Appends a new row to the convergence log CSV."""
        try:
            with open(self.log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation, best_aocc, best_name])
        except IOError as e:
            print(f"Warning: Failed to write to convergence log {self.log_filename}. Error: {e}")

    def _sort_key(self, ind):
        """Sort key for individuals: high score first, errors last."""
        if ind['error']:
            return -float('inf')  # Errors are always worst
        return ind['score']

    def run(self):
        """
        Executes the LLaMEA evolutionary process.
        """
        
        # --- Initialization (Generation 0) ---
        print(f"--- Initialization: Generating {self.mu} initial parents ---")
        prompt = self._get_initial_prompt()
        last_generated = None

        for i in range(self.mu):
            print(f"Initializing parent {i+1}/{self.mu}...")
            if last_generated:
                # Use the last generated individual as context for the next
                prompt = self._build_offspring_prompt(last_generated)
            
            response = self.gemini_caller.call_gemini(prompt)
            name, code = self._extract_code(response)

            if not code:
                print("Could not extract code. Adding a placeholder individual.")
                individual = {"name": "InitFail", "code": "", "score": 0.0, "error": "Code extraction failed."}
            else:
                score, error = self._save_and_evaluate(name, code)
                print(f"Generated '{name}' with score: {score}")
                individual = {"name": name, "code": code, "score": score, "error": error}
            
            self.population.append(individual)
            self.history.append(individual)
            last_generated = individual

        # Sort initial population
        self.population = sorted(self.population, key=self._sort_key, reverse=True)
        
        best_init_score = self.population[0]['score'] if self.population else 0.0
        best_init_name = self.population[0]['name'] if self.population else "InitFail"
        print(f"Initialization complete. Best score: {best_init_score:.4f}")

        # Log generation 0
        self._log_convergence(0, best_init_score, best_init_name)

        # --- Generational Loop ---
        for gen in range(self.generations):
            print(f"\n--- Generation {gen+1}/{self.generations} ---")
            offspring_population = []

            # Check if population is empty (which can happen in 'comma' strategy if mu=0 or all init failed)
            if not self.population:
                print("Population is empty. Cannot generate offspring. Stopping run.")
                break
            
            for i in range(self.lambda_):
                # Select a parent (e.g., round-robin)
                parent = self.population[i % len(self.population)]
                print(f"Generating offspring {i+1}/{self.lambda_} (parent: {parent['name']})...")
                
                prompt = self._build_offspring_prompt(parent)
                response = self.gemini_caller.call_gemini(prompt)
                name, code = self._extract_code(response)

                if not code:
                    print("Could not extract code. Adding a placeholder individual.")
                    individual = {"name": "OffspringFail", "code": "", "score": 0.0, "error": "Code extraction failed."}
                else:
                    score, error = self._save_and_evaluate(name, code)
                    print(f"Generated '{name}' with score: {score}")
                    individual = {"name": name, "code": code, "score": score, "error": error}
                
                offspring_population.append(individual)
                self.history.append(individual)
            
            # --- Selection ---
            self.population = self._select_next_generation(offspring_population)
            
            best_gen_score = self.population[0]['score'] if self.population else 0.0
            best_gen_name = self.population[0]['name'] if self.population else "SelectFail"
            print(f"Generation {gen+1} complete. Best score: {best_gen_score:.4f} ({best_gen_name})")

            # Log this generation's best score
            self._log_convergence(gen + 1, best_gen_score, best_gen_name)

        print("\nEvolution finished!")
        
        # Save the best algorithm from the final population
        if not self.population:
             print("No successful algorithms were generated in the entire run.")
             with open("best_algorithm.py", "w") as f:
                f.write("# No successful algorithm was generated.")
             return

        best_algo = self.population[0] # The best is the first in the sorted list
        
        print(f"Best algorithm found: '{best_algo['name']}' with score: {best_algo['score']}")
        with open("best_algorithm.py", "w") as f:
            if best_algo['code']:
                f.write(best_algo['code'])
                print("Best algorithm saved to best_algorithm.py")
            else:
                f.write("# Best algorithm had no code (e.g., init failure).")
                print("Best algorithm code was empty.")