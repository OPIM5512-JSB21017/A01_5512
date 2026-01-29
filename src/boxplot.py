from pathlib import Path
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Boxplot for median house value (as a DataFrame)
plt.figure(figsize=(6, 5))
df[["MedHouseVal"]].boxplot()
plt.title("California Housing Dataset – Median House Value")
plt.ylabel("Median House Value ($100,000s)")
plt.tight_layout()

# Ensure figs exists
figs_dir = Path(__file__).resolve().parents[1] / "figs"
figs_dir.mkdir(parents=True, exist_ok=True)

# Save the figure
output_path = figs_dir / "boxplot.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Boxplot saved to {output_path}")
