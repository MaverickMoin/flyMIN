import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Updated Mapping
df_baseline = pd.read_csv('cent_simulation/auth_simulation_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_simulation_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_simulation_results.csv')
df_GPU_latency = pd.read_csv('data/GPU_70B_latency.csv')

prefill_size = 512
decoding_list = [128, 512, 1024, 3584]
transformer_blocks = 80

gpu_prefill = df_GPU_latency[(df_GPU_latency['Phase'] == 'Prefill')]['Latency (min)'].iloc[0]
gpu_decoding = [df_GPU_latency[(df_GPU_latency['Phase'] == f'Decoding_{d}')]['Latency (min)'].iloc[0] for d in decoding_list]

def get_cent_data(df):
    df_filtered = df[(df['Model'] == "Llama2-70B") & (df['Device number'] == 32) & 
                     (df['Pipeline parallelism'] == transformer_blocks) & (df['Tensor parallelism'] == 1)]
    
    prefill = df_filtered[df_filtered['Sequence length'] <= prefill_size]['Token latency (ms)'].mean() * prefill_size / 1000 / 60
    decoding = []
    for d in decoding_list:
        dec_lat = df_filtered[(df_filtered['Sequence length'] > prefill_size) & 
                              (df_filtered['Sequence length'] <= prefill_size + d)]['Token latency (ms)'].mean() * d / 1000 / 60
        decoding.append(dec_lat)
    return [prefill]*4, decoding

pref_b, dec_b = get_cent_data(df_baseline)
pref_m, dec_m = get_cent_data(df_mount)
pref_f, dec_f = get_cent_data(df_fly)

x_labels = ["Out 128", "Out 512", "Out 1k", "Out 3.5k"]
x = np.arange(len(x_labels))
w = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

def plot_bar(x_pos, pref, dec, label, color_pref, color_dec):
    ax.bar(x_pos, pref, width=w, color=color_pref, edgecolor='black', label=f"{label} (Prefill)")
    ax.bar(x_pos, dec, width=w, bottom=pref, color=color_dec, edgecolor='black', label=f"{label} (Decoding)")

plot_bar(x - 1.5*w, [gpu_prefill]*4, gpu_decoding, "GPU", "lightgray", "gray")
plot_bar(x - 0.5*w, pref_b, dec_b, "CENT Baseline", "lightblue", "dodgerblue")
plot_bar(x + 0.5*w, pref_m, dec_m, "mountMIN", "lightgreen", "forestgreen")
plot_bar(x + 1.5*w, pref_f, dec_f, "flyMIN", "lightsalmon", "firebrick")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylabel("Latency (min)", fontsize=12)
ax.set_title("Prefill + Decoding Latency (Input 512)", fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.5)

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig('figures/figure_14d_comparison.pdf', bbox_inches='tight')
