import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_QoS_file

# Load the data for all three configurations
df_baseline = pd.read_csv('cent_simulation/auth_processed_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_processed_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_processed_results.csv')

batch = [1, 2, 4, 8, 16, 80]
seqlen = 4096
num_devices = 32

def get_cent_data(df):
    latency = []      # minutes
    throughput = []   # queries per minute
    
    for pp in batch[:-1]:
        f_df = df[(df['Model'] == 'Llama2-70B') & (df['Device number'] == 32) & (df['Seqlen'] == seqlen) & (df['Pipeline parallelism'] == pp) & (df['Tensor parallelism'] == num_devices // pp) & (df['Phase'] == 'end2end')]
        if not f_df.empty:
            latency.append(f_df['Total Latency (s)'].mean().item() / 60)
            throughput.append(f_df['Throughput (tokens/s)'].mean().item() * 60 / seqlen)

    # 80 PP case handled separately
    f_df = df[(df['Model'] == 'Llama2-70B') & (df['Device number'] == 32) & (df['Seqlen'] == seqlen) & (df['Pipeline parallelism'] == 80) & (df['Tensor parallelism'] == 1) & (df['Phase'] == 'end2end')]
    if not f_df.empty:
        latency.append(f_df['Total Latency (s)'].mean().item() / 60)
        throughput.append(f_df['Throughput (tokens/s)'].mean().item() * 60 / seqlen)
        
    return latency, throughput

# Extract data for all architectures
dict_baseline = {}
dict_baseline["latency"], dict_baseline["throughput"] = get_cent_data(df_baseline)

dict_mount = {}
dict_mount["latency"], dict_mount["throughput"] = get_cent_data(df_mount)

dict_fly = {}
dict_fly["latency"], dict_fly["throughput"] = get_cent_data(df_fly)

font = 20
dict_GPU_70B = load_QoS_file("data/GPU_70B_4k.csv")
plt.figure(figsize=(10, 8))
    
# Plot all 4 lines
plt.plot(dict_baseline["throughput"], dict_baseline["latency"], marker='s', linestyle='-', color='Red', label="CENT Baseline")
plt.plot(dict_mount["throughput"], dict_mount["latency"], marker='^', linestyle='-', color='lightgreen', label="mountMIN")
plt.plot(dict_fly["throughput"], dict_fly["latency"], marker='D', linestyle='-', color='salmon', label="flyMIN")
plt.plot(dict_GPU_70B["throughput"], dict_GPU_70B["latency"], marker='o', linestyle='-', color='Blue', label="GPU Baseline")

plt.legend(loc="upper left", fontsize=font)
plt.tick_params(axis='x', labelsize=font)
plt.tick_params(axis='y', labelsize=font)

plt.xlabel('Throughput (Query/min)', fontsize=font)
plt.ylabel('Query Latency (min)', fontsize=font)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14b.pdf')

# Prepare data export to CSV
# Calculate max length to pad missing rows with NaN safely
max_len = max(len(dict_baseline['latency']), len(dict_mount['latency']), 
              len(dict_fly['latency']), len(dict_GPU_70B['latency']))

def pad(lst, length):
    return lst + [np.nan] * (length - len(lst))

df_export = pd.DataFrame({
    'Baseline Throughput (Query/min)': pad(dict_baseline['throughput'], max_len),
    'Baseline Latency (min)': pad(dict_baseline['latency'], max_len),
    'mountMIN Throughput (Query/min)': pad(dict_mount['throughput'], max_len),
    'mountMIN Latency (min)': pad(dict_mount['latency'], max_len),
    'flyMIN Throughput (Query/min)': pad(dict_fly['throughput'], max_len),
    'flyMIN Latency (min)': pad(dict_fly['latency'], max_len),
    'GPU Throughput (Query/min)': pad(dict_GPU_70B['throughput'], max_len),
    'GPU Latency (min)': pad(dict_GPU_70B['latency'], max_len)
})

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df_export.to_csv('figure_source_data/figure_14b.csv', index=False)

