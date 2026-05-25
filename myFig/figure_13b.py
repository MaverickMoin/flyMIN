import os
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np

# Updated Mapping (Data Sources)
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')   # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')       # flyMIN
df_GPU_throughput = pd.read_csv('data/GPU_throughput.csv')
devices = {
    'Llama2-7B': 8,
    'Llama2-13B': 20,
    'Llama2-70B': 32,
}

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
phases = ['prefill', 'decoding', 'end2end']

# ORIGINAL logic: Using transformer blocks for Pipeline Parallelism
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}
seqlen = 4096

speedup_baseline, speedup_mount, speedup_fly = [], [], []

for phase in phases:
    for model in models:
        # ORIGINAL calculation logic
        gpu_tp = df_GPU_throughput[(df_GPU_throughput['Model'] == model)][phase].iloc[0]
        
        def get_speedup(df):
            # Filtering uses Pipeline parallelism = transformer_block[model] and Tensor parallelism = 1
            filtered = df[(df['Model'] == model) & 
                          (df['Seqlen'] == seqlen) & (df['Device number'] == devices[model]) & 
                          (df['Pipeline parallelism'] == transformer_block[model]) & (df['Tensor parallelism'] == 1) & 
                          #(df['Pipeline parallelism'] == 1) & (df['Tensor parallelism'] == devices[model]) & 
                          (df['Phase'] == phase)]
            return filtered['Throughput (tokens/s)'].iloc[0] / gpu_tp

        speedup_baseline.append(get_speedup(df_baseline))
        speedup_mount.append(get_speedup(df_mount))
        speedup_fly.append(get_speedup(df_fly))

# ORIGINAL logic: Geomean calculated strictly on the last 3 values (End-to-End phase)
speedup_baseline.append(gmean(speedup_baseline[-3:]))
speedup_mount.append(gmean(speedup_mount[-3:]))
speedup_fly.append(gmean(speedup_fly[-3:]))


# --- PLOTTING (Retaining the 2-layer x-labels and styles) ---

# LAYER 1: Model Sizes
x_labels_layer1 = ['7B', '13B', '70B', 
                   '7B', '13B', '70B', 
                   '7B', '13B', '70B', 'Geomean']

x = np.arange(len(x_labels_layer1))
width = 0.25

# Using subplots to easily reference the axes (ax)
fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(x - width, speedup_baseline, width, color='gray', edgecolor='black', label='CENT Baseline')
ax.bar(x, speedup_mount, width, color='dodgerblue', edgecolor='black', label='mountMIN')
ax.bar(x + width, speedup_fly, width, color='firebrick', edgecolor='black', label='flyMIN')

# Apply Layer 1 labels directly to the ticks
ax.set_xticks(x)
ax.set_xticklabels(x_labels_layer1, fontsize=18)
ax.tick_params(axis='x', pad=5) # Adds a little padding so text doesn't touch the axis

# Increase Y-axis tick label font size
ax.tick_params(axis='y', labelsize=18)

# LAYER 2: Group Categories
# Using a transform where x is in data coordinates and y is in axes coordinates
trans = ax.get_xaxis_transform()

# Place text centered at x=1 (middle of first 3), x=4 (middle of next 3), x=7 (middle of last 3)
# y=-0.14 pushes the text below the first layer of labels
ax.text(1, -0.14, 'Prefill', ha='center', va='top', transform=trans, fontsize=20)
ax.text(4, -0.14, 'Decode', ha='center', va='top', transform=trans, fontsize=20)
ax.text(7, -0.14, 'Overall', ha='center', va='top', transform=trans, fontsize=20)

ax.set_ylabel('CENT/GPU Normalized Throughput', fontsize=20)
ax.set_title('Throughput Comparison (Tokens/s)', fontsize=22)

# Add dividers
ax.axvline(x=2.5, color='gray', linestyle='--')
ax.axvline(x=5.5, color='gray', linestyle='--')
ax.axvline(x=8.5, color='gray', linestyle='--')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Increase Legend font size
ax.legend(fontsize=18)

# Add extra space at the bottom so the second layer isn't cut off
plt.subplots_adjust(bottom=0.2)

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_13b_comparison_PP.pdf', bbox_inches='tight')
plt.close()

# --- NEW CALCULATION AND PRINT BLOCK ---
print("\nThroughput Improvement compared to Baseline ((My Throughput - Baseline) / Baseline):")

# Generate structured labels for the 10 data points
labels_detailed = [
    'Prefill - Llama2-7B', 'Prefill - Llama2-13B', 'Prefill - Llama2-70B',
    'Decode - Llama2-7B', 'Decode - Llama2-13B', 'Decode - Llama2-70B',
    'Overall - Llama2-7B', 'Overall - Llama2-13B', 'Overall - Llama2-70B',
    'Overall - Geomean'
]

for i, label in enumerate(labels_detailed):
    base_speedup = speedup_baseline[i]
    mount_speedup = speedup_mount[i]
    fly_speedup = speedup_fly[i]
    
    if base_speedup > 0:
        mount_improvement = (mount_speedup - base_speedup) / base_speedup
        fly_improvement = (fly_speedup - base_speedup) / base_speedup
    else:
        mount_improvement = 0
        fly_improvement = 0
        
    print(f"{label}:")
    print(f"  mountMIN: {mount_improvement * 100:.2f}%")
    print(f"  flyMIN: {fly_improvement * 100:.2f}%")