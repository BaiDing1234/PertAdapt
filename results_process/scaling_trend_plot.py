import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib as mpl
from scipy import stats

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.unicode_minus': False
})


# x-axis (parameter sizes in millions)
x = np.log(np.array([3, 10, 100]))
n = 5
datasets = ['Norman', 'Replogle RPE1']
methods = ['AIDO.Cell', 'AIDO.Cell+Ours']

with open('/home/ding.bai/ding-scFoundation/scFoundation-main/PertAdapter/result_process/pkl/scaling_trend.pkl', 'rb') as f:
    results_dict = pickle.load(f)



fig, axs = plt.subplots(1,2)
fig.set_size_inches(7.5, 3)

for j, dataset in enumerate(datasets):
    
    ax = axs[j]
    # baseline
    mse_base = results_dict[dataset]['AIDO.Cell']['mean'] # mean
    std_base = results_dict[dataset]['AIDO.Cell']['std']  # std
    CI_base = std_base * stats.t.ppf(0.975, n-1)/np.sqrt(n)


    # ours
    mse_ours = results_dict[dataset]['AIDO.Cell+Ours']['mean'] # mean
    std_ours = results_dict[dataset]['AIDO.Cell+Ours']['std']  # std
    CI_ours = std_ours * stats.t.ppf(0.975, n-1)/np.sqrt(n)


    if dataset == 'Norman':
        dataset_title = 'Norman (Overall)'
        ax.set_ylim(((mse_ours.min()//0.02)*0.02, (mse_base.max()//0.02 + 1.1)*0.02))
        ax.set_yticks(np.arange((mse_ours.min()//0.02)*0.02, (mse_base.max()//0.02 + 2)*0.02, step=0.02))
    else:
        dataset_title = dataset
        ax.set_ylim(((mse_ours.min()//0.02)*0.02, (mse_base.max()//0.02 + 0.5)*0.02))
        ax.set_yticks(np.arange((mse_ours.min()//0.02)*0.02, (mse_base.max()//0.02 + 1)*0.02, step=0.02))
    ax.grid(axis='y', alpha=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # baseline
    ax.plot(x, mse_base, marker='.', label="AIDO.Cell", color="#0072B2")
    ax.fill_between(x, mse_base-CI_base, mse_base+CI_base, alpha=0.2, color="#0072B2")
    ax.errorbar(x, mse_base, yerr=CI_base, color="#0072B2", capsize=1)


    # ours
    ax.plot(x, mse_ours, marker='.', label="AIDO.Cell+Ours", color="#CC79A7")
    ax.fill_between(x, mse_ours-CI_ours, mse_ours+CI_ours, alpha=0.2, color="#CC79A7")
    ax.errorbar(x, mse_ours, yerr=CI_ours, color="#CC79A7", capsize=1)

    # x ticks as 3M, 10M, 100M
    ax.set_xticks(x, ["3M", "10M", "100M"])

    ax.set_xlabel("log FM Size (Params)")
    ax.set_ylabel(f"{dataset_title} MSE")
    #plt.title("Scaling trend of MSE vs FM size")
    ax.legend(frameon=True, loc='upper right')
    #ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"fig/scaling_trend.svg", format="svg", dpi=300)
fig.savefig(f"fig/scaling_trend.pdf", format="pdf", dpi=300)
