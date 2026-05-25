import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_simulation_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_simulation_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_simulation_results.csv')

parallel_list = ["PP=80 TP=1", "PP=16 TP=2", "PP=8 TP=4", "PP=4 TP=8", "PP=2 TP=16", "PP=1 TP=32"]
configs = [(80,1), (16,2), (8,4), (4,8), (2,16), (1,32)]
seqlen = 4096
transformer_blocks = 80

def get_breakdowns(df):
    pim, cxl, acc = [], [], []
    for pp, tp in configs:
        filt = df[(df['Model'] == 'Llama2-70B') & (df['Device number'] == 32) & 
                  (df['Pipeline parallelism'] == pp) & (df['Tensor parallelism'] == tp)]
        if not filt.empty:
            pim.append(filt['PIM latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
            cxl.append(filt['CXL latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
            acc.append(filt['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
        else:
            pim.append(0); cxl.append(0); acc.append(0)
    return np.array(pim), np.array(cxl), np.array(acc)

pim_b, cxl_b, acc_b = get_breakdowns(df_baseline)
pim_m, cxl_m, acc_m = get_breakdowns(df_mount)
pim_f, cxl_f, acc_f = get_breakdowns(df_fly)

y = np.arange(len(parallel_list))
height = 0.25

fig, ax = plt.subplots(figsize=(10, 8))

def plot_stacked(y_offsets, pim, cxl, acc, hatch=''):
    ax.barh(y_offsets, pim, height, color="navajowhite", edgecolor='black', hatch=hatch)
    ax.barh(y_offsets, cxl, height, left=pim, color="lightblue", edgecolor='black', hatch=hatch)
    ax.barh(y_offsets, acc, height, left=pim+cxl, color="darkgreen", edgecolor='black', hatch=hatch)

plot_stacked(y + height, pim_b, cxl_b, acc_b, hatch='')    # Baseline
plot_stacked(y, pim_m, cxl_m, acc_m, hatch='//')           # mountMIN
plot_stacked(y - height, pim_f, cxl_f, acc_f, hatch='xx')  # flyMIN

import matplotlib.patches as mpatches
c1 = mpatches.Patch(color='navajowhite', label='PIM')
c2 = mpatches.Patch(color='lightblue', label='CXL')
c3 = mpatches.Patch(color='darkgreen', label='Acc')
c4 = mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='CENT Baseline')
c5 = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='mountMIN')
c6 = mpatches.Patch(facecolor='white', edgecolor='black', hatch='xx', label='flyMIN')
ax.legend(handles=[c1, c2, c3, c4, c5, c6], loc='upper right', fontsize=10)

ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels(parallel_list)
ax.set_xlabel("Latency (min)")
ax.set_title("Latency Breakdown Comparison")

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_14c_comparison.pdf', bbox_inches='tight')
