import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_simulation_results_long_context.csv')
df_mount = pd.read_csv('cent_simulation/mount_simulation_results_long_context.csv')
df_fly = pd.read_csv('cent_simulation/fly_simulation_results_long_context.csv')
gpu_decoding = pd.read_csv('data/GPU_70B_decoding.csv')

seqlen_list = [4096, 8192, 16384, 32768]
adjusted_seqlen = [(s + s - 3584) // 2 for s in seqlen_list]
gpu_throughput_list = gpu_decoding['Throughput (tokens/s)'].tolist()

speedup_baseline, speedup_mount, speedup_fly = [], [], []

for i, seqlen in enumerate(adjusted_seqlen):
    def get_speedup(df):
        df_filtered = df[(df['Model'] == 'Llama2-70B') & (df['Device number'] == 32) & 
                         (df['Sequence length'] == seqlen) & (df['Pipeline parallelism'] == 80) & 
                         (df['Tensor parallelism'] == 1)]
        return df_filtered['Throughput (tokens/s)'].iloc[0] / gpu_throughput_list[i]

    speedup_baseline.append(get_speedup(df_baseline))
    speedup_mount.append(get_speedup(df_mount))
    speedup_fly.append(get_speedup(df_fly))

context_lengths = ["4K", "8K", "16K", "32K"]
x = np.arange(len(context_lengths))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, speedup_baseline, width, color='skyblue', edgecolor='black', label='Baseline')
plt.bar(x, speedup_mount, width, color='lightgreen', edgecolor='black', label='mountMIN')
plt.bar(x + width, speedup_fly, width, color='salmon', edgecolor='black', label='flyMIN')

plt.xticks(x, context_lengths, fontsize=12)
plt.xlabel("Context Length", fontsize=12)
plt.ylabel("CENT / GPU Speedup", fontsize=12)
plt.title("Long Context Speedup (PP=80 TP=1)", fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_14a_comparison.pdf', bbox_inches='tight')
