import os
from llamea import LLaMEA

def main():
    """
    Main function to run the LLaMEA experiment.
    """
    # It is recommended to set the API key as an environment variable
    # for security reasons.
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        return

    # Configuration for the LLaMEA run
    BUDGET = 1000
    DIMS = 5
    GENERATIONS = 15      # Total number of generations
    MU = 2                # Parent population size
    LAMBDA = 3            # Offspring population size
    SELECTION_STRATEGY = 'plus' # 'plus' for (mu+lambda), 'comma' for (mu,lambda)

    # Initialize and run the LLaMEA framework
    llamea_instance = LLaMEA(api_key=gemini_api_key,
                             budget=BUDGET,
                             dims=DIMS,
                             generations=GENERATIONS,
                             mu=MU,
                             lambda_=LAMBDA,
                             selection_strategy=SELECTION_STRATEGY,
                             llm_model='gemini-2.5-pro')
    llamea_instance.run()

if __name__ == "__main__":
    main()