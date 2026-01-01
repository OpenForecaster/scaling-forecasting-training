import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch

# --- Data (in exact order as shown in the chart) ---
universities = [
    "Toronto", "Waterloo", "Hopkins", "UPenn", "Berkeley",
    "MIT", "CalTech", "Cornell", "Stanford", "CMU",
    "WashU", "Harvard", "Princeton", "MPI-IS / ELLIS \nTübingen"
]

gpus_per_student = [
    0.0000, 0.0000, 0.0050, 0.0100, 0.0250,
    0.0400, 0.0450, 0.0800, 0.1400, 0.2200,
    0.2900, 0.3300, 0.7800, 4.2110
]

# --- Create the plot ---
plt.figure(figsize=(10, 6))
bars = plt.bar(universities, gpus_per_student, color='#4285F4', edgecolor='none')

# Make bar corners rounded (top corners only)
# ax = plt.gca()
# for bar in bars:
#     # Get the rectangle properties
#     x, y = bar.get_xy()
#     width = bar.get_width()
#     height = bar.get_height()
    
#     # Create rounded rectangle with rounded top corners only
#     # Using round4 (rounds all corners) with small radius for subtle rounding
#     rounded_rect = FancyBboxPatch(
#         (x, y), width, height,
#         boxstyle="round,pad=0.02",
#         edgecolor='none',
#         facecolor=bar.get_facecolor(),
#         zorder=bar.get_zorder(),
#         transform=ax.transData
#     )
    
#     # Remove original rectangle and add rounded one
#     bar.remove()
#     ax.add_patch(rounded_rect)

# Title - left aligned
plt.title("GPUs/Students vs. University", fontsize=28, color='gray', pad=20, loc='left')

# Axis labels
# plt.xlabel("University", fontsize=18)
plt.ylabel("GPUs/Students", fontsize=18, labelpad=15)

# X-axis: rotate labels 45 degrees
plt.xticks(rotation=45, ha='right', fontsize=14)

# Note: We'll shift the last label after tight_layout

# Y-axis: set ticks and format with 4 decimal places

# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=14)
plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0], fontsize=14)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))
# plt.ylim(0, 0.8)

ax = plt.gca()

# Grid: horizontal lines only, light gray (behind bars)
ax.set_axisbelow(True)  # Put grid and axes behind plot elements
plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=1, alpha=1)
plt.grid(axis='x', visible=False)

# Background
ax.set_facecolor('white')
plt.gcf().set_facecolor('white')

# Spines: only bottom and right visible, remove top and left
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Shift the last x-tick label to the right (for the newline label)
# Do this after tight_layout so positions are finalized
tick_labels = ax.get_xticklabels()
if tick_labels:
    last_label = tick_labels[-1]
    last_label_text = last_label.get_text()
    
    # Get the current position of the label
    label_x, label_y = last_label.get_position()
    
    # Remove the original label by making it invisible
    last_label.set_visible(False)
    
    # Add a manually positioned text annotation with controlled offset
    # ADJUST offset_x to control how much to shift right (in data coordinates)
    # Positive values shift right, negative shift left
    # Typical range: 0.1 (small) to 0.5 (large)
    offset_x = 0.4  # <-- ADJUST THIS VALUE to control the shift amount
    offset_y = - 0.3
    
    # Use the same transform as the original tick label
    ax.text(
        label_x + offset_x,  # x position: original position + offset
        label_y + offset_y,  # y position: same as original label
        last_label_text,
        rotation=45,
        ha='right',  # Keep same alignment as other labels
        va='baseline',  # Baseline align to match other labels
        fontsize=14,
        transform=last_label.get_transform()  # Use same transform as tick labels
    )

# Save the plot
plt.savefig("gpuquota.png", dpi=400, bbox_inches='tight', facecolor='white')

# Show the plot
plt.show()
