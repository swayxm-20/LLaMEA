# LLaMEA: Large Language Model Evolutionary Algorithm

Welcome to **LLaMEA**, a framework that leverages Large Language Models (LLMs) to automatically design, evaluate, and evolve novel metaheuristic algorithms for continuous optimization problems.

## Overview

LLaMEA stands for **Large Language Model Evolutionary Algorithm**. Instead of manually designing optimization algorithms (like Genetic Algorithms or Particle Swarm Optimization), LLaMEA uses an LLM (such as Google's Gemini or Groq) as a mutation and crossover operator to iteratively write, evaluate, and refine optimization algorithms in Python. 

The framework evaluates the generated algorithms against the **[IOHexperimenter](https://iohprofiler.github.io/)** (IOH) benchmark suite, specifically the BBOB suite, to calculate their Area Over the Convergence Curve (AOCC) and drive the evolutionary process.

## Key Features

- **Automated Algorithm Design**: Generates complete Python classes for optimization algorithms via LLM prompting.
- **Evolutionary Loop**: Implements an evolution strategy—supporting both `(mu + lambda)` and `(mu, lambda)` selection strategies.
- **Robust Benchmarking**: Seamlessly integrates with the IOHexperimenter suite to rigorously evaluate algorithms across multiple dimensions and benchmark problems.
- **Extensible LLM Integration**: Comes with support for Google Gemini (`gemini-2.5-pro` by default) and includes a caller for Groq.
- **Convergence Logging**: Automatically logs the generation-wise progression of the best algorithm's performance into CSV files.

## Project Structure

- `main.py`: The main entry point to configure and launch the LLaMEA experiment.
- `llamea.py`: Contains the core `LLaMEA` class which orchestrates the evolutionary loop (Initialization, Generation, Evaluation, Selection).
- `ioh_runner.py`: Handles the execution of generated algorithms on the IOH benchmark suite and calculates fitness (AOCC).
- `gemini_caller.py` / `groq_caller.py`: Wrappers around the respective LLM APIs.
- `random_search.py`: An example algorithm provided to the LLM as a few-shot prompt/baseline.
- `requirements.txt`: Project dependencies.

## Prerequisites

- Python 3.8+
- An API key for Google Gemini (or Groq, if configured).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/swayxm-20/LLaMEA.git
   cd LLaMEA
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set your Gemini API key as an environment variable:
   - **Linux/macOS:**
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```
   - **Windows (Command Prompt):**
     ```cmd
     set GEMINI_API_KEY="your-api-key-here"
     ```
   - **Windows (PowerShell):**
     ```powershell
     $env:GEMINI_API_KEY="your-api-key-here"
     ```

2. Run the main experiment script:
   ```bash
   python main.py
   ```

3. **Check Results:**
   - Generated algorithms will be saved as `.py` files inside the `generated_algos/` directory.
   - Evolution logs will be saved in a CSV file (e.g., `log_mu2_lambda3_plus_YYYYMMDD_HHMMSS.csv`).
   - The overall best algorithm will be saved as `best_algorithm.py` in the root directory.

## How It Works

1. **Initialization:** The system prompts the LLM to generate `mu` distinct parent algorithms based on a system prompt and an example (`random_search.py`).
2. **Evaluation:** Each generated code snippet is executed against the IOH suite. Its convergence performance is normalized into a fitness score (AOCC).
3. **Reproduction & Mutation:** The LLM is prompted to mutate or combine parent algorithms to produce `lambda` offspring. The prompt includes performance histories and error messages of previous algorithms to guide the LLM.
4. **Selection:** The top algorithms are selected to form the next generation.
5. **Termination:** After the specified number of generations, the evolutionary run concludes, and the best discovered algorithm is outputted.

## License

This project is open-source and available for research and educational purposes.
