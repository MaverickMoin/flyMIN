import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Updated Mapping for the 3 setups
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')

batch = [1, 2, 5, 10, 20, 40]
seqlen = 4096
num_devices = 20

def get_curve_data(df):
    latencies, throughputs = [], []
    for pp in batch[:-1]:
        filtered = df[(df['Model'] == 'Llama2-13B') & (df['Device number'] == num_devices) & 
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
tp_mount, lat_mount = get_curve_data(df_mount)
tp_fly, lat_fly = get_curve_data(df_fly)

# --- Normalization ---
# x: divide by max baseline throughput  → scale [0, 1]
# y: divide by min baseline latency     → scale starts at 1.0, rises proportionally
tp_max  = max(tp_base)
lat_min = min(lat_base)

def normalize(tp, lat):
    return [t / tp_max for t in tp], [l / lat_min for l in lat]

tp_base_n,  lat_base_n  = normalize(tp_base,  lat_base)
tp_mount_n, lat_mount_n = normalize(tp_mount, lat_mount)
tp_fly_n,   lat_fly_n   = normalize(tp_fly,   lat_fly)

plt.figure(figsize=(8, 6))
font = 12

# Plot the 3 configurations
plt.plot(tp_base_n,  lat_base_n,  marker='s', linestyle='-', color='dimgray',       label="CENT Baseline")
plt.plot(tp_mount_n, lat_mount_n, marker='^', linestyle='-', color='cornflowerblue', label="mountMIN")
plt.plot(tp_fly_n,   lat_fly_n,   marker='D', linestyle='-', color='darkred',        label="flyMIN")

# Shade improvement gap between baseline and flyMIN
y_shared = np.linspace(min(min(lat_base_n), min(lat_fly_n)),
                       max(max(lat_base_n), max(lat_fly_n)), 200)
plt.fill_betweenx(
    y_shared,
    np.interp(y_shared, lat_base_n,  tp_base_n),
    np.interp(y_shared, lat_fly_n,   tp_fly_n),
    alpha=0.12, color='darkred', label='Improvement gap'
)

# Annotate throughput improvement % at each flyMIN point
for xb, yb, xf, yf in zip(tp_base_n, lat_base_n, tp_fly_n, lat_fly_n):
    pct = (xf - xb) / xb * 100
    plt.annotate(f'+{pct:.1f}%', xy=(xf, yf), xytext=(5, 2),
                 textcoords='offset points', fontsize=8, color='darkred')

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc="upper left", fontsize=font)
plt.xlabel('Normalized Throughput (fraction of max)', fontsize=font)
plt.ylabel('Normalized Latency (× baseline min)', fontsize=font)
plt.title('Latency vs Throughput (Llama2-13B, 20 devices)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

if not os.path.exists("figures"):
    os.mkdir("figures")
plt.savefig('figures/figure_14b_norm_comparison_13B.pdf', bbox_inches='tight')