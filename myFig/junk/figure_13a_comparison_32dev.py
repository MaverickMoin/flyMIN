import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')   # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')       # flyMIN

df_GPU_latency = pd.read_csv('data/GPU_latency.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
devices = {'Llama2-7B': 8, 'Llama2-13B': 20, 'Llama2-70B': 32}
transformer_block = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}
target_devices = 32
seqlen = 4096

speedup_baseline, speedup_mount, speedup_fly = [], [], []

for model in models:
    # Get baseline GPU latency
    gpu_lat = df_GPU_latency[(df_GPU_latency['Model'] == model)]['End-to-end Latency (s)'].iloc[0]
    
    def get_speedup(df):
        # Filter on Model, Seqlen, and exactly 32 Devices
        filtered = df[(df['Model'] == model) & (df['Seqlen'] == seqlen) & 
                      (df['Device number'] == target_devices) & 
                      (df['Pipeline parallelism'] == transformer_block[model]) & (df['Tensor parallelism'] == 1) & 
                      (df['Phase'] == 'end2end')]
        if filtered.empty:
            return 0
        # If multiple PP/TP topologies exist for 32 devices, pick the best latency
        min_lat = filtered['Total Latency (s)'].min()
        return gpu_lat / min_lat

    speedup_baseline.append(get_speedup(df_baseline))
    speedup_mount.append(get_speedup(df_mount))
    speedup_fly.append(get_speedup(df_fly))

# Filter out 0s for gmean if data was missing
speedup_baseline.append(gmean([s for s in speedup_baseline if s > 0]))
speedup_mount.append(gmean([s for s in speedup_mount if s > 0]))
speedup_fly.append(gmean([s for s in speedup_fly if s > 0]))

x_labels = models + ['Geomean']
x = np.arange(len(x_labels))
width = 0.25 

plt.figure(figsize=(10, 5))
plt.bar(x - width, speedup_baseline, width, color='skyblue', edgecolor='black', label='Baseline')
plt.bar(x, speedup_mount, width, color='lightgreen', edgecolor='black', label='mountMIN')
plt.bar(x + width, speedup_fly, width, color='salmon', edgecolor='black', label='flyMIN')

plt.xticks(x, x_labels, fontsize=12)
plt.ylabel('GPU/CENT Normalized Latency', fontsize=12)
plt.title(f'End-to-end Latency Comparison (Devices={target_devices})', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_13a_comparison_32dev.pdf', bbox_inches='tight')
plt.close()
