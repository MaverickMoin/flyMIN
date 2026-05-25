import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Updated Mapping for the 3 setups
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')

batch = [1, 2, 4, 8, 16, 32]
seqlen = 4096
num_devices = 32

def get_curve_data(df):
    latencies, throughputs = [], []
    for pp in batch[:-1]:
        filtered = df[(df['Model'] == 'Llama2-70B') & (df['Device number'] == 32) & 
                      (df['Seqlen'] == seqlen) & (df['Pipeline parallelism'] == pp) & 
                      (df['Tensor parallelism'] == num_devices // pp) & (df['Phase'] == 'end2end')]
        if not filtered.empty:
            latencies.append(filtered['Total Latency (s)'].mean().item() / 60)
            throughputs.append(filtered['Throughput (tokens/s)'].mean().item() * 60 / seqlen)
    
    # Sorting to ensure the lines draw smoothly without zig-zags
    sorted_pairs = sorted(zip(throughputs, latencies))
    if sorted_pairs:
        throughputs, latencies = zip(*sorted_pairs)
    return list(throughputs), list(latencies)

tp_base, lat_base = get_curve_data(df_baseline)
tp_fly, lat_fly = get_curve_data(df_fly)

plt.figure(figsize=(8, 6))
font = 18

# Plot the 2 configurations
plt.plot(tp_base, lat_base, marker='s', linestyle='-', color='dimgray', label="CENT Baseline")
plt.plot(tp_fly,  lat_fly,  marker='D', linestyle='-', color='darkred',  label="flyMIN")

# Shade improvement gap: interpolate tp values over a shared latency grid
y_shared = np.linspace(min(min(lat_base), min(lat_fly)),
                       max(max(lat_base), max(lat_fly)), 200)
plt.fill_betweenx(
    y_shared,
    np.interp(y_shared, lat_base, tp_base),  # baseline throughput at each latency
    np.interp(y_shared, lat_fly,  tp_fly),   # flyMIN throughput at each latency
    alpha=0.12, color='darkred', label='Improvement'
)

# Annotate throughput improvement % at each data point
for xb, yb, xf, yf in zip(tp_base, lat_base, tp_fly, lat_fly):
    pct = (xf - xb) / xb * 100
    plt.annotate(f'+{pct:.1f}%', xy=(xf, yf), xytext=(5, 2),
                 textcoords='offset points', fontsize=8, color='darkred')
plt.tick_params(axis='y', labelsize=font)
plt.legend(fontsize=16)
plt.legend(loc="upper left", fontsize=font)
plt.xlabel('Throughput (Query/min)', fontsize=font)
plt.ylabel('Query Latency (min)', fontsize=font)
plt.title('Latency vs Throughput (Llama2-70B, 32 devices)', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)

if not os.path.exists("figures"):
    os.mkdir("figures")
plt.savefig('figures/figure_14b_comparison_70B.pdf', bbox_inches='tight')