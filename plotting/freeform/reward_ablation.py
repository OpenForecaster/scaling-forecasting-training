
# Recreate the plots with hardcoded values and save data files + images.
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import textwrap
import matplotlib as mpl
import scienceplots
import os
mpl.style.use(['science'])

# -----------------------
# Hardcoded data (read off the figure)
# -----------------------

iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Left panel — Accuracy (%)
acc_blue  = [19.3, 22.1, 23, 21.8, 23.4, 26.0, 27.0, 26.4, 26.6, 26.9, 27.2]  # Only Accuracy
acc_red   = [19.3, 20.8, 24.0, 24.0, 23.9, 23.2, 23.7, 26.0, 24.8, 25.4, 26.6]  # Only Freeform Brier
acc_purp  = [19.3, 21.6, 24.8, 26.1, 27.1, 24.8, 25.5, 26.3, 27.0, 27.5, 25.6]  # Accuracy + Freeform Brier

# Right panel — Freeform Brier Score
brier_blue = [0.000, -0.015, -0.030, -0.045, -0.055, -0.070, -0.065, -0.060, -0.040, -0.020, -0.018]  # Only Accuracy
brier_red  = [0.000,  0.075,  0.092,  0.101,  0.103,  0.094,  0.108,  0.123,  0.118,  0.121,  0.120]  # Only Freeform Brier
brier_purp = [0.000,  0.060,  0.104,  0.102,  0.116,  0.108,  0.106,  0.105,  0.103,  0.118,  0.106]  # Accuracy + Freeform Brier

# Define good shades of red, blue, and purple (purple is a mix of red and blue)
red_color   = '#E24A33'  # vibrant red
green_color = '#47A23F'  # vibrant green
blue_color  = '#348ABD'  # vibrant blue
# Mix red and blue for a visually balanced purple
purple_color = '#9B5FC0'  # vibrant purple (mix of red and blue)


# Only plot till 900 iterations
k = 10
iterations = iterations[:k]
acc_blue = acc_blue[:k]
acc_red = acc_red[:k]
acc_purp = acc_purp[:k]
brier_blue = brier_blue[:k]
brier_red = brier_red[:k]
brier_purp = brier_purp[:k]

# -----------------------
# Plot
# -----------------------

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "serif",
})


# fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
ax1, ax2 = axes

# Left: Accuracy
l1, = ax1.plot(iterations, acc_blue,  "-o", linewidth=4, markersize=10, color=blue_color, alpha=0.8, label="Only Accuracy")
l2, = ax1.plot(iterations, acc_red,   "-o", linewidth=4, markersize=10, color=red_color, alpha=0.8,       label="Only Brier")
l3, = ax1.plot(iterations, acc_purp,  "-o", linewidth=4, markersize=10, color=purple_color, alpha=0.8,    label="Accuracy + Brier")
ax1.set_xlabel("Training Iterations")
ax1.set_ylabel("Accuracy (\%)", labelpad=12)
ax1.set_xlim(-10, max(iterations) + 10)
ax1.set_ylim(19.0, 28.5)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='both', labelsize=20)
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)

# Right: Freeform Brier Score
ax2.plot(iterations, brier_blue, "-o", linewidth=4, markersize=10, color=blue_color, alpha=0.8)
ax2.plot(iterations, brier_red,  "-o", linewidth=4, markersize=10, color=red_color, alpha=0.8)
ax2.plot(iterations, brier_purp, "-o", linewidth=4, markersize=10, color=purple_color, alpha=0.8)
ax2.set_xlabel("Training Iterations")
ax2.set_ylabel("Brier Score \ (higher is better)", labelpad=0)
ax2.set_xlim(-10, max(iterations) + 10)
ax2.set_ylim(-0.075, 0.13)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='both', labelsize=20)

# One shared legend at the very top
fig.legend(handles=[l1, l2, l3], loc="upper center", ncol=3, frameon=True, fancybox=True, fontsize=18, bbox_to_anchor=(0.5, 1.1))


plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

# Save (optional)
fig_path = "plots/reward_ablation/training_objectives_recreated.png"
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path, dpi=300)
# also save as a pdf
fig.savefig(fig_path.replace(".png", ".pdf"), dpi=300)
plt.close(fig)

plt.show()


# plt.tight_layout()
# plt.subplots_adjust(wspace=0.18)

# combined_path = "plots/lineplots/data_filtering/data_filtering_training_progression_recreated.png"
# os.makedirs(os.path.dirname(combined_path), exist_ok=True)
# plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
# # save this also as a pdf
# plt.savefig(combined_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight', facecolor='white')
# plt.close(fig)
