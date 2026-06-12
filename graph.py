import pandas as pd
import matplotlib.pyplot as plt

# List of your CSV file paths
csv_files = [ 
    "./(2+3)-gpt-oss-120b/log_mu2_lambda3_plus_20251102_165239.csv",
    "./(2+3)-gemini-2.5-pro/log_mu2_lambda3_plus_20251102_231456.csv"
]

# Optional: labels for the plot legend
labels = [
    "(2+3)-gpt-oss-120b",
    "(2+3)-gemini-2.5-pro"
]

plt.figure(figsize=(10, 6))

# Read and plot each CSV
for file, label in zip(csv_files, labels):
    df = pd.read_csv(file)
    plt.plot(df["generation"], df["best_aocc"], label=label)

plt.title("Comparison of Best AOCC Across Generations")
plt.xlabel("Generation")
plt.ylabel("Best AOCC")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save or show the figure
plt.savefig("aocc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
