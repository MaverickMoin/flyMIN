import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the three distinct sources for ROPE
df_baseline = pd.read_csv('cent_simulation/baseline_ropeprocessed_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_ropeprocessed_results.csv')   # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_ropeprocessed_results.csv')       # flyMIN

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
phases = ['prefill', 'decoding', 'end2end']
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}
devices = {
    'Llama2-7B': 8,
    'Llama2-13B': 20,
    'Llama2-70B': 32,
}
seqlen = 4096

tp_baseline, tp_mount, tp_fly = [], [], []

for phase in phases:
    for model in models:
        def get_throughput(df):
            filtered = df[(df['Model'] == model) & (df['Seqlen'] == seqlen) & (df['Device number'] == devices[model]) & 
                          (df['Pipeline parallelism'] == transformer_block[model]) & 
                          (df['Tensor parallelism'] == 1) & (df['Phase'] == phase)]
            
            columns_to_check = ['Model', 'Device number', 'Seqlen', 'Pipeline parallelism', 'Tensor parallelism', 'Phase']
            print(filtered[columns_to_check])

            if not filtered.empty:
                # Divide by 1000 to scale to 1000 tokens/sec
                return filtered['Throughput (tokens/s)'].iloc[0] / 1000.0
            return 0

        tp_baseline.append(get_throughput(df_baseline))
        tp_mount.append(get_throughput(df_mount))
        tp_fly.append(get_throughput(df_fly))

x_labels = ['7B\nPrefill', '13B\nPrefill', '70B\nPrefill',
            '7B\nDecoding', '13B\nDecoding', '70B\nDecoding',
            '7B\nEnd-to-end', '13B\nEnd-to-end', '70B\nEnd-to-end']

x = np.arange(len(x_labels))
width = 0.25

# Plotting
plt.figure(figsize=(14, 6))
plt.bar(x - width, tp_baseline, width, color='darkgray', edgecolor='black', label='Baseline')
plt.bar(x, tp_mount, width, color='royalblue', edgecolor='black', label='mountMIN')
plt.bar(x + width, tp_fly, width, color='firebrick', edgecolor='black', label='flyMIN')

plt.xticks(x, x_labels, fontsize=11)
# Updated y-axis label and title
plt.ylabel('Throughput (1000 Tokens/s)', fontsize=12)
plt.title('ROPE Throughput Comparison (1000 Tokens/s)', fontsize=14)
plt.axvline(x=2.5, color='gray', linestyle='--')
plt.axvline(x=5.5, color='gray', linestyle='--')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

if not os.path.exists("figures"): 
    os.mkdir("figures")
plt.savefig('figures/rope_figure_13b_comparison.pdf', bbox_inches='tight')
plt.close()

# Export CSV
df_export = pd.DataFrame({
    'Config': [f"{m} {p}" for p in phases for m in models],
    'Baseline Throughput': tp_baseline,
    'mountMIN Throughput': tp_mount,
    'flyMIN Throughput': tp_fly
})
if not os.path.exists("figure_source_data"): 
    os.mkdir("figure_source_data")
df_export.to_csv('figure_source_data/rope_figure_13b_comparison.csv', index=False)