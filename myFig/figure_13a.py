import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')         # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')       # flyMIN

df_GPU_latency = pd.read_csv('data/GPU_latency.csv')

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
devices = {'Llama2-7B': 8, 'Llama2-13B': 20, 'Llama2-70B': 32}
seqlen = 4096
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}

speedup_baseline, speedup_mount, speedup_fly = [], [], []

for model in models:
    gpu_lat = df_GPU_latency[(df_GPU_latency['Model'] == model)]['End-to-end Latency (s)'].iloc[0]
    
    def get_speedup(df):
        filtered = df[(df['Model'] == model) & (df['Seqlen'] == seqlen) & 
                      (df['Pipeline parallelism'] == 1) & (df['Tensor parallelism'] == devices[model]) & 
                      #(df['Pipeline parallelism'] == transformer_block[model]) & (df['Tensor parallelism'] == 1) & 
                      (df['Phase'] == 'end2end')]
        return gpu_lat / filtered['Total Latency (s)'].iloc[0]

    speedup_baseline.append(get_speedup(df_baseline))
    speedup_mount.append(get_speedup(df_mount))
    speedup_fly.append(get_speedup(df_fly))

speedup_baseline.append(gmean(speedup_baseline))
speedup_mount.append(gmean(speedup_mount))
speedup_fly.append(gmean(speedup_fly))

x_labels = models + ['Geomean']
x = np.arange(len(x_labels))
width = 0.25 

plt.figure(figsize=(10, 5))
plt.bar(x - width, speedup_baseline, width, color='slategray', edgecolor='black', label='CENT Baseline')
plt.bar(x, speedup_mount, width, color='cornflowerblue', edgecolor='black', label='mountMIN')
plt.bar(x + width, speedup_fly, width, color='darkred', edgecolor='black', label='flyMIN')

font=18
plt.tick_params(axis='y', labelsize=font)
plt.xticks(x, x_labels, fontsize=font)
plt.ylabel('GPU/CENT Normalized Latency', fontsize=font)
plt.title('End-to-end Latency Comparison', fontsize=font+2)
plt.tick_params(axis='y', labelsize=font)
plt.legend(fontsize=font)
plt.grid(axis='y', linestyle='--', alpha=0.7)

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_13a_comparison.pdf', bbox_inches='tight')

# Calculate and print latency improvements (can be derived from speedups: reduction = 1 - (base_speedup / new_speedup))
print("\nLatency Reduction compared to Baseline ((Baseline Latency - My Latency) / Baseline Latency):")
for i, label in enumerate(x_labels):
    base_speedup = speedup_baseline[i]
    mount_speedup = speedup_mount[i]
    fly_speedup = speedup_fly[i]
    
    if mount_speedup > 0:
        mount_improvement = 1 - (base_speedup / mount_speedup)
    else:
        mount_improvement = 0
        
    if fly_speedup > 0:
        fly_improvement = 1 - (base_speedup / fly_speedup)
    else:
        fly_improvement = 0
        
    print(f"{label}:")
    print(f"  mountMIN: {mount_improvement * 100:.2f}%")
    print(f"  flyMIN: {fly_improvement * 100:.2f}%")