from ioh_runner import IOHRunner
from matplotlib import pyplot as plt

import traceback

if __name__ == '__main__':
    runner = IOHRunner(budget=1000, dims=5)

    try:
        score1 = runner.evaluate("./CMA-ES.py")
        score2 = runner.evaluate("./DE.py")
        score3 = runner.evaluate("./PSO.py")
        score4 = 0.5302936238314312

        labels = ["CMA-ES", "DE", "PSO", "AMDEOBL"]
        scores = [score1, score2, score3, score4]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, scores, width=0.6)
        plt.title("Comparison of AOCC with standard Algorithms")
        plt.xlabel("Algorithms")
        plt.ylabel("AOCC")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Save or show the figure
        plt.savefig("graph3.png", dpi=300, bbox_inches="tight")
        plt.show()

    except Exception as e:
        # --- Enhanced Error Logging ---
        # Capture the full traceback to get the exact line number and context of the error.
        error_message = traceback.format_exc()
        print(error_message)