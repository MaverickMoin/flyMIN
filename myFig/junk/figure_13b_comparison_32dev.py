import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')   # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')       # flyMIN
df_GPU_throughput = pd.read_csv('data/GPU_throughput.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
phases = ['prefill', 'decoding', 'end2end']
transformer_block = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
target_devices = 32
seqlen = 4096

speedup_baseline, speedup_mount, speedup_fly = [], [], []

for phase in phases:
    for model in models:
        gpu_tp = df_GPU_throughput[(df_GPU_throughput['Model'] == model)][phase].iloc[0]
        
        def get_speedup(df):
            # Filter strictly by 32 Devices
            filtered = df[(df['Model'] == model) & (df['Seqlen'] == seqlen) & 
                          (df['Device number'] == target_devices) & 
                          (df['Pipeline parallelism'] == transformer_block[model]) & (df['Tensor parallelism'] == 1) & 
                          (df['Phase'] == phase)]
            if filtered.empty:
                return 0
            # Extract the best throughput for the 32 device budget
            max_tp = filtered['Throughput (tokens/s)'].max()
            return max_tp / gpu_tp

        speedup_baseline.append(get_speedup(df_baseline))
        speedup_mount.append(get_speedup(df_mount))
        speedup_fly.append(get_speedup(df_fly))

speedup_baseline.append(gmean(speedup_baseline[-3:]))
speedup_mount.append(gmean(speedup_mount[-3:]))
speedup_fly.append(gmean(speedup_fly[-3:]))

x_labels = ['7B\nPrefill', '13B\nPrefill', '70B\nPrefill',
            '7B\nDecoding', '13B\nDecoding', '70B\nDecoding',
            '7B\nEnd-to-end', '13B\nEnd-to-end', '70B\nEnd-to-end', 'Geomean']

x = np.arange(len(x_labels))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width, speedup_baseline, width, color='skyblue', edgecolor='black', label='Baseline')
plt.bar(x, speedup_mount, width, color='lightgreen', edgecolor='black', label='mountMIN')
plt.bar(x + width, speedup_fly, width, color='salmon', edgecolor='black', label='flyMIN')

plt.xticks(x, x_labels, fontsize=11)
plt.ylabel('CENT/GPU Normalized Throughput', fontsize=12)
plt.title(f'Throughput Comparison (Tokens/s) - Devices={target_devices}', fontsize=14)
plt.axvline(x=2.5, color='gray', linestyle='--')
plt.axvline(x=5.5, color='gray', linestyle='--')
plt.axvline(x=8.5, color='gray', linestyle='--')
plt.legend()

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_13b_comparison_32dev.pdf', bbox_inches='tight')
plt.close()
