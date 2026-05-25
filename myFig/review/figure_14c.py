import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

# Attempt to import InOut_latency, default to 0 if not available for standalone execution
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from cent_simulation.utils import InOut_latency
except ImportError:
    InOut_latency = 0  

def get_factors(n):
    return sorted([i for i in range(1, n + 1) if n % i == 0])

# Load the three distinct sources
df_baseline = pd.read_csv('cent_simulation/auth_simulation_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_simulation_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_simulation_results.csv')

seqlen = 4096

# Hardcoded Target Configuration
model = 'Llama2-13B'
device = 20
transformer_blocks = 40

def extract_latencies(df_source, model, device, transformer_blocks, tps):
    PIM, CXL, Acc, CPU = [], [], [], [] 
    
    # Base configuration: PP = transformer_blocks, TP = 1
    df = df_source[(df_source['Model'] == model) & (df_source['Device number'] == device) & 
                   (df_source['Pipeline parallelism'] == transformer_blocks) & (df_source['Tensor parallelism'] == 1)]
    if df.empty:
        PIM.append(0); CXL.append(0); Acc.append(0); CPU.append(0)
    else:
        PIM.append(df['PIM latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
        CXL.append(df['CXL latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
        Acc.append(df['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen) 
        CPU.append(InOut_latency * seqlen / 1000 / 60)

    # TP factors
    for tp in tps:
        if tp == 1: continue
        pp = device // tp
        df = df_source[(df_source['Model'] == model) & (df_source['Device number'] == device) & 
                       (df_source['Pipeline parallelism'] == pp) & (df_source['Tensor parallelism'] == tp)]
        if df.empty:
            PIM.append(0); CXL.append(0); Acc.append(0); CPU.append(0)
            continue

        PIM.append(df['PIM latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
        CXL.append(df['CXL latency'].mean() * transformer_blocks / 1000 / 60 * seqlen)
        Acc.append(df['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen) 
        CPU.append(InOut_latency * seqlen / 1000 / 60)
        
    return np.array(PIM), np.array(CXL), np.array(Acc), np.array(CPU) 

tps = get_factors(device)
pps = [device // tp for tp in tps]

# Build dynamic labels matching logic
parallel_list = [f"PP={transformer_blocks}"]
for pp, tp in zip(pps, tps):
    if tp == 1: continue 
    parallel_list.append(f"PP={pp} TP={tp}")

# Extract latencies for all three datasets
pim_b, cxl_b, acc_b, cpu_b = extract_latencies(df_baseline, model, device, transformer_blocks, tps) 
pim_m, cxl_m, acc_m, cpu_m = extract_latencies(df_mount, model, device, transformer_blocks, tps)
pim_f, cxl_f, acc_f, cpu_f = extract_latencies(df_fly, model, device, transformer_blocks, tps)

# Output Data to CSV
df_results = pd.DataFrame({'Config': parallel_list,
    'Baseline PIM': pim_b, 'Baseline CXL': cxl_b, 'Baseline Acc': acc_b, 'Baseline CPU': cpu_b, 
    'Mount PIM': pim_m, 'Mount CXL': cxl_m, 'Mount Acc': acc_m, 'Mount CPU': cpu_m,
    'Fly PIM': pim_f, 'Fly CXL': cxl_f, 'Fly Acc': acc_f, 'Fly CPU': cpu_f
})

if not os.path.exists("figure_source_data"): os.mkdir("figure_source_data")
df_results.to_csv(f'figure_source_data/figure_{model}_{device}_14c.csv', index=False)

# Graph Generation
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(parallel_list))
bar_height = 0.25 

sources = [
    ("CENT Baseline", pim_b, cxl_b, acc_b, cpu_b, y_pos - bar_height, ''),
    ("mountMIN", pim_m, cxl_m, acc_m, cpu_m, y_pos, '///'), 
    ("flyMIN", pim_f, cxl_f, acc_f, cpu_f, y_pos + bar_height, '\\\\\\') 
]

for label, pim, cxl, acc, cpu, pos, hatch in sources:
    ax.barh(pos, pim, height=bar_height, color="navajowhite", edgecolor='black', hatch=hatch, label="") 
    ax.barh(pos, cxl, left=pim, height=bar_height, color="lightblue", edgecolor='black', hatch=hatch, label="")
    ax.barh(pos, acc, left=pim+cxl, height=bar_height, color="darkgreen", edgecolor='black', hatch=hatch, label="") 
    ax.barh(pos, cpu, left=pim+cxl+acc, height=bar_height, color="dimgray", edgecolor='black', hatch=hatch, label="") 

ax.set_yticks(y_pos)
ax.set_yticklabels(parallel_list, fontsize=12)
ax.set_xlabel("Query Latency (minute)", fontsize=12)
ax.set_title(f"Latency Breakdown: {model} on {device} Devices")

pim_patch = mpatches.Patch(facecolor='navajowhite', edgecolor='black', label='PIM')
cxl_patch = mpatches.Patch(facecolor='lightblue', edgecolor='black', label='CXL')
acc_patch = mpatches.Patch(facecolor='darkgreen', edgecolor='black', label='PNM')
cpu_patch = mpatches.Patch(facecolor='dimgray', edgecolor='black', label='Host CPU')
baseline_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='CENT Baseline')
mountmin_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='mountMIN') 
flymin_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='\\\\\\', label='flyMIN') 

legend_patches = [pim_patch, cxl_patch, acc_patch, cpu_patch, baseline_hatch, mountmin_hatch, flymin_hatch]

ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

if not os.path.exists("figures"): os.mkdir("figures")
plt.savefig(f'figures/figure_{model}_{device}_14c.pdf')
plt.close()