import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 0. Load your data from the specified path
csv_path = "/home/mhchu/AI-Companion/human-ai/OpenAI_moderation/results/human_ai_chatlogs.csv"
df = pd.read_csv(csv_path)

# 1. Identify numeric category score columns (excluding 'turn')
category_cols = df.select_dtypes(include='number').columns.tolist()
if 'turn' in category_cols:
    category_cols.remove('turn')

# 2. Grid size for plotting
n = len(category_cols)
ncols = 4
nrows = math.ceil(n / ncols)

# 3. Create subplots for distributions
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes_flat = axes.flatten()

for ax, cat in zip(axes_flat, category_cols):
    vals = df[cat].dropna()
    hist, bins = np.histogram(vals, bins=50, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(centers, hist)
    ax.set_title(cat)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')

# Hide any unused subplots
for ax in axes_flat[n:]:
    ax.set_visible(False)

plt.tight_layout()

# 4. Save the plot grid as PNG
png_path = "category_distributions.png"
plt.savefig(png_path, dpi=300)
print(f"Saved plot grid to {png_path}")

# 5. Compute descriptive statistics
stats = df[category_cols].describe()

# 6. Compute counts above thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
counts = pd.DataFrame(
    {f">{thr}": (df[category_cols] > thr).sum() for thr in thresholds}
)

# 7. Write stats and counts to text file
txt_path = "category_stats_and_counts.txt"
with open(txt_path, "w") as f:
    f.write("Descriptive Statistics:\n")
    f.write(stats.to_string())
    f.write("\n\nCounts of rows above thresholds:\n")
    f.write(counts.to_string())
print(f"Saved descriptive statistics and threshold counts to {txt_path}")


