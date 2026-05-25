import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# Load the three distinct sources for ROPE simulation
df_baseline = pd.read_csv('cent_simulation/baseline_ropesimulation_results.csv') # Baseline
df_mount = pd.read_csv('cent_simulation/mount_ropesimulation_results.csv')       # mountMIN
df_fly = pd.read_csv('cent_simulation/fly_ropesimulation_results.csv')           # flyMIN

models = ['Llama2-7B', 'Llama2-13B', 'Llama2-70B']
devices = {
    'Llama2-7B': 8,
    'Llama2-13B': 20,
    'Llama2-70B': 32,
}
transformer_block = {
    'Llama2-7B': 32,
    'Llama2-13B': 40,
    'Llama2-70B': 80,
}
seqlen = 4096

lat_baseline, lat_mount, lat_fly = [], [], []

for model in models:
    def get_latency(df):
        # Simulation files sometimes use 'Sequence length' instead of 'Seqlen'
        seq_col = 'Seqlen' if 'Seqlen' in df.columns else 'Sequence length'
        
        # Safely evaluate the sequence condition before combining with bitwise operators
        seq_condition = (df[seq_col] == seqlen) if seq_col in df.columns else True
        
        filtered = df[(df['Model'] == model) & (df['Device number'] == devices[model]) & 
                      seq_condition & 
                      (df['Pipeline parallelism'] == transformer_block[model]) & (df['Tensor parallelism'] == 1) ]
                      #(df['Pipeline parallelism'] == 1) & (df['Tensor parallelism'] == devices[model]) ]
        
        # If 'Phase' exists in these CSVs, filter for end2end
        if 'Phase' in df.columns:
            phase_filtered = filtered[filtered['Phase'] == 'end2end']
            if not phase_filtered.empty:
                filtered = phase_filtered

        if not filtered.empty:
            # CALCULATE LATENCY USING THE REQUESTED FORMULA AND CONVERT TO MICROSECONDS
            pim_lat = filtered['PIM latency'].iloc[0]
            acc_lat = filtered['Acc latency'].iloc[0]
            return (pim_lat + acc_lat) * 1000
        
        return 0.0

    lat_baseline.append(get_latency(df_baseline))
    lat_mount.append(get_latency(df_mount))
    lat_fly.append(get_latency(df_fly))


# --- Calculate and append the Geomean ---
lat_baseline.append(gmean(lat_baseline))
lat_mount.append(gmean(lat_mount))
lat_fly.append(gmean(lat_fly))

# Update labels and spacing to include Geomean
x_labels = models + ['Geomean']
x = np.arange(len(x_labels))
width = 0.25 

# Plotting
# Reduced height to 6 (from 8)
plt.figure(figsize=(10, 6)) 
plt.bar(x - width, lat_baseline, width, color='darkgray', edgecolor='black', label='Baseline')
plt.bar(x, lat_mount, width, color='royalblue', edgecolor='black', label='mountMIN')
plt.bar(x + width, lat_fly, width, color='firebrick', edgecolor='black', label='flyMIN')

font=18
plt.tick_params(axis='y', labelsize=font)
plt.xticks(x, x_labels, fontsize=font)
plt.ylabel('RoPE Latency (microseconds)', fontsize=font)
plt.title('ROPE Latency Comparision', fontsize=font+2)

# Dynamically increase the y-axis limit by 30% to give the legend plenty of headroom
if max(lat_baseline) > 0:
    plt.ylim(0, max(lat_baseline) * 1.3)

plt.legend(fontsize=font-1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

if not os.path.exists("figures"): 
    os.mkdir("figures")
plt.savefig('figures/rope_simulation_comparison.pdf', bbox_inches='tight')
plt.close()

# Export CSV (Updated to use x_labels)
df_export = pd.DataFrame({
    'Model': x_labels,
    'Baseline Latency': lat_baseline,
    'mountMIN Latency': lat_mount,
    'flyMIN Latency': lat_fly
})
if not os.path.exists("figure_source_data"): 
    os.mkdir("figure_source_data")
df_export.to_csv('figure_source_data/rope_simulation_comparison_PP.csv', index=False)

# --- CALCULATION AND PRINT BLOCK ---
print("\nLatency Reduction compared to Baseline ((Baseline - My Latency) / Baseline):")
for i, label in enumerate(x_labels):
    base_lat = lat_baseline[i]
    mount_lat = lat_mount[i]
    fly_lat = lat_fly[i]
    
    if base_lat > 0:
        mount_improvement = (base_lat - mount_lat) / base_lat
        fly_improvement = (base_lat - fly_lat) / base_lat
    else:
        mount_improvement = 0.0
        fly_improvement = 0.0
        
    print(f"{label}:")
    print(f"  mountMIN: {mount_improvement * 100:.2f}%")
    print(f"  flyMIN:   {fly_improvement * 100:.2f}%")