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
# Hardcoded data (estimated from the provided figure)
# -----------------------

iterations = [0, 50, 100, 150, 200, 250, 300, 350]

# Accuracy (%) — three methods
acc_purple = [19.4, 21.2, 21.1, 21.9, 22.3, 21.6, 22.3, 22.1]  # Filtering with leakage removal
acc_red    = [19.4, 20.1, 20.1, 20.0, 21.0, 20.7, 21.2, 20.9]  # Leakage removal but no filtering
acc_blue   = [19.3, 17.2, 14.6, 16.6, 17.4, 16.2, 15.8, 16.5]  # No filtering
acc_blue   = [19.3, 17.2, 15.6, 16.6, 17.4, 16.2, 15.8, 16.5]  # No filtering

# Freeform Brier Score — three methods
brier_purple = [0.000, 0.069, 0.070, 0.082, 0.086, 0.087, 0.097, 0.092]
brier_red    = [0.000, 0.062, 0.064, 0.066, 0.073, 0.071, 0.080, 0.081]
brier_blue   = [0.000, -0.005, -0.025, -0.024, -0.034, -0.040, -0.072, -0.037]
brier_blue   = [0.000, -0.005, -0.025, -0.024, -0.034, -0.040, -0.05, -0.037]

red_color = '#FF0000'
blue_color = '#0000FF'
green_color = '#00FF00'

# actually choose good red, green, blue shades
# More vibrant/better red, green, blue shades (colorblind-friendly)
red_color   = '#E24A33'  # vibrant red
green_color = '#47A23F'  # vibrant green
blue_color  = '#348ABD'  # vibrant blue

# -----------------------
# Save data to CSV files
# -----------------------

acc_df = pd.DataFrame({
    "Training Iterations": iterations,
    "All filters (green)": acc_purple,
    "Leakage filter (blue)": acc_red,
    "No filter (red)": acc_blue,
})
brier_df = pd.DataFrame({
    "Training Iterations": iterations,
    "All filters (green)": brier_purple,
    "Leakage filter (blue)": brier_red,
    "No filter (red)": brier_blue,
})

# acc_csv_path = "/mnt/data/accuracy_hardcoded.csv"
# brier_csv_path = "/mnt/data/brier_hardcoded.csv"
# acc_df.to_csv(acc_csv_path, index=False)
# brier_df.to_csv(brier_csv_path, index=False)

# -----------------------
# Plot styling helpers
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

def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# -----------------------
# Combined plot: Accuracy and Freeform Brier Score
# -----------------------

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# # Accuracy subplot
# ax1.plot(iterations, acc_purple, '-o', linewidth=4, markersize=10, label="All filters", color=green_color, alpha=0.8)
# ax1.plot(iterations, acc_red,    '-o', linewidth=4, markersize=10, label="Leakage filter only", color=blue_color, alpha=0.8)
# ax1.plot(iterations, acc_blue,   '-o', linewidth=4, markersize=10, label="No filter", color=red_color, alpha=0.8)
# # ax1.set_xlim(-5, 355)
# # ax1.set_ylim(14.4, 22.6)
# ax1.set_xlabel('Training Iterations', fontsize=20, fontweight='bold', labelpad=8)
# ax1.set_ylabel('Accuracy (\%)', fontsize=20, fontweight='bold', labelpad=12)
# ax1.grid(True, alpha=0.3, linestyle='--')
# ax1.tick_params(axis='both', labelsize=20)
# # dashed reference line and annotation "3x less data"
# ax1.hlines(21.2, xmin=100, xmax=300, colors='black', linestyles=(0, (5,5)), linewidth=2)
# ax1.text(10, 21.5, '3x less data', fontsize=22, ha='left', va='bottom')

# # Brier subplot
# ax2.plot(iterations, brier_purple, '-o', linewidth=4, markersize=10, label="All filters", color=green_color, alpha=0.8)
# ax2.plot(iterations, brier_red,    '-o', linewidth=4, markersize=10, label="Leakage filter only", color=blue_color, alpha=0.8)
# ax2.plot(iterations, brier_blue,   '-o', linewidth=4, markersize=10, label="No filter", color=red_color, alpha=0.8)
# # ax2.set_xlim(-5, 355)
# # ax2.set_ylim(-0.075, 0.10)
# ax2.set_xlabel('Training Iterations', fontsize=20, fontweight='bold', labelpad=8)
# ax2.set_ylabel('Brier Score', fontsize=20, fontweight='bold', labelpad=-4)
# ax2.grid(True, alpha=0.3, linestyle='--')
# ax2.tick_params(axis='both', labelsize=20)
# # dashed reference line and annotation "2x faster"
# ax2.hlines(0.080, xmin=160, xmax=300, colors='black', linestyles=(0, (5,5)), linewidth=2)
# ax2.text(100, 0.087, '2x faster', fontsize=22, ha='left', va='bottom')

# # Legend above both subplots
# handles, labels = ax1.get_legend_handles_labels()
# handles = handles[::-1]
# labels = labels[::-1]
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(labels), fontsize=18, frameon=True, fancybox=True)

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.18)

# combined_path = "plots/lineplots/data_filtering/data_filtering_training_progression_recreated.png"
# os.makedirs(os.path.dirname(combined_path), exist_ok=True)
# plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
# # save this also as a pdf
# plt.savefig(combined_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight', facecolor='white')
# plt.close(fig)



fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

# Accuracy subplot
ax1.plot(iterations, acc_purple, '-o', linewidth=4, markersize=10, label="All filters", color=green_color, alpha=0.8)
ax1.plot(iterations, acc_red,    '-o', linewidth=4, markersize=10, label="Leakage filter only", color=blue_color, alpha=0.8)
ax1.plot(iterations, acc_blue,   '-o', linewidth=4, markersize=10, label="No filter", color=red_color, alpha=0.8)
# ax1.set_xlim(-5, 355)
# ax1.set_ylim(14.4, 22.6)
ax1.set_xlabel('Training Iterations', fontsize=20, fontweight='bold', labelpad=8)
ax1.set_ylabel('Accuracy (\%)', fontsize=20, fontweight='bold', labelpad=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='both', labelsize=20)
# dashed reference line and annotation "3x less data"
ax1.hlines(21.2, xmin=100, xmax=300, colors='black', linestyles=(0, (5,5)), linewidth=2)
ax1.text(10, 21.5, '3x faster', fontsize=22, ha='left', va='bottom')

# Legend above both subplots
handles, labels = ax1.get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.56, 1.06), ncol=len(labels), fontsize=14, frameon=True, fancybox=True)

plt.tight_layout()

combined_path = "plots/lineplots/data_filtering/data_filtering_training_progression_recreated_accuracy.png"
os.makedirs(os.path.dirname(combined_path), exist_ok=True)
plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
# save this also as a pdf
plt.savefig(combined_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)