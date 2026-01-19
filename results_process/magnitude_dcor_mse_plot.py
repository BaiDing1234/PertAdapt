import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

# Set Arial globally in Matplotlib configuration
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5

df = pd.read_csv("csv/norman_magnitude_mse.csv")
n = 5
methods = df["method"].tolist()
x = np.arange(len(methods))

mag_mse = df["Magnitude MSE"].tolist()
mag_std = df["Magnitude MSE std"].tolist()
dcorr_mse = df["Dcorr MSE"].tolist()
dcorr_std = df["Dcorr MSE std"].tolist()

mag_CI = np.array(mag_std) * stats.t.ppf(0.975, n-1)/np.sqrt(n)
dcorr_CI = np.array(dcorr_std) * stats.t.ppf(0.975, n-1)/np.sqrt(n)

# Wong colorblind-safe RGB palette (最适合科研，可配 Illustrator)
WONG_COLORS = [
    "#78C0A4",  # random
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow         scFoundation
    "#0072B2",  # blue           AIDO.Cell
    "#D55E00",  # vermillion     scFoundation+Ours
    "#CC79A7"   # reddish purple AIDO.Cell+Ours
]


colors = WONG_COLORS[:len(methods)]

import matplotlib.pyplot as plt
import numpy as np

def beautify_bar_plot(ax):
    # Horizontal light grid
    ax.grid(axis='y', alpha=0.25)

    # Remove top/right borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Thinner axis lines (optional but nicer)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Tick font size
    ax.tick_params(axis='both', labelsize=8)


# ---- Magnitude MSE ----
fig, axs = plt.subplots(1,2)
fig.set_size_inches(7.5, 3.5)
ax = axs[0]
ax.bar(x, mag_mse, yerr=mag_CI, capsize=4, color=colors, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right", va="center", rotation_mode="anchor")
ax.set_ylabel("Magnitude MSE", fontsize=9)
ax.set_ylim((0.4, 0.91))
ax.set_yticks(np.arange(0.4, 1.0, step=0.1))

beautify_bar_plot(ax)


# ---- Dcorr MSE ----
ax = axs[1]
ax.bar(x, dcorr_mse, yerr=dcorr_CI, capsize=4, color=colors, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right", va="center", rotation_mode="anchor")
ax.set_ylabel("Model fit MSE", fontsize=9)
ax.set_ylim((0.05, 0.101))
ax.set_yticks(np.arange(0.05, 0.11, step=0.01))

beautify_bar_plot(ax)
fig.tight_layout()
fig.savefig("fig/magnitude_dcorr_mse.svg", format="svg", dpi=300)

print("[Saved] SVG figures in RGB mode")
