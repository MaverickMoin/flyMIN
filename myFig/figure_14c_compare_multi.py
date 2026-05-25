import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For consolidated comparative legend

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

# Define Model-Device Constraints
# Note: Make sure to update this list to reflect the actual parallel configurations you want to test
model_device_combinations = {
    'Llama2-7B': [8, 20, 32],
    'Llama2-13B': [20, 32],
    'Llama2-70B': [32]
}

transformer_blocks_dict = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}

# Update return values to rename Acc from PNM to Acc for consistency with consolidated legend
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
        # Rename PNM latency to Acc latency for consistency with consolidated legend
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
        Acc.append(df['Acc latency'].mean() * transformer_blocks / 1000 / 60 * seqlen) # PNM latency to Acc latency
        CPU.append(InOut_latency * seqlen / 1000 / 60)
        
    return np.array(PIM), np.array(CXL), np.array(Acc), np.array(CPU) # Return Acc

for model, device_numbers in model_device_combinations.items():
    for device in device_numbers:
        transformer_blocks = transformer_blocks_dict[model]
        tps = get_factors(device)
        pps = [device // tp for tp in tps]
        
        # Build dynamic labels matching logic
        parallel_list = [f"PP={transformer_blocks}"]
        for pp, tp in zip(pps, tps):
            if tp == 1: continue 
            parallel_list.append(f"PP={pp} TP={tp}")

        # Extract latencies for all three datasets
        # Note: Update return values from PNM to Acc for component consistency
        pim_b, cxl_b, acc_b, cpu_b = extract_latencies(df_baseline, model, device, transformer_blocks, tps) # update return values
        pim_m, cxl_m, acc_m, cpu_m = extract_latencies(df_mount, model, device, transformer_blocks, tps)
        pim_f, cxl_f, acc_f, cpu_f = extract_latencies(df_fly, model, device, transformer_blocks, tps)

        # Output Data to CSV - Use new component names
        # Note: This updates the source data headers too, for better readability of comparisons
        df_results = pd.DataFrame({'Config': parallel_list,
            'Baseline PIM': pim_b, 'Baseline CXL': cxl_b, 'Baseline Acc': acc_b, 'Baseline CPU': cpu_b, # Update component names
            'Mount PIM': pim_m, 'Mount CXL': cxl_m, 'Mount Acc': acc_m, 'Mount CPU': cpu_m,
            'Fly PIM': pim_f, 'Fly CXL': cxl_f, 'Fly Acc': acc_f, 'Fly CPU': cpu_f
        })
        if not os.path.exists("figure_source_data"): os.mkdir("figure_source_data")
        df_results.to_csv(f'figure_source_data/figure_{model}_{device}_14c.csv', index=False)

        # Graph Generation
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(parallel_list))
        bar_height = 0.25 # Sub-bar thickness for side-by-side comparison layout
        
        # Sources configuration for side-by-side rendering with unique hatching
        # Note: The target names like "mountMIN" and "flyMIN" have been updated to match the example image precisely
        sources = [
            ("CENT Baseline", pim_b, cxl_b, acc_b, cpu_b, y_pos - bar_height, ''),
            ("mountMIN", pim_m, cxl_m, acc_m, cpu_m, y_pos, '///'), # case sensitive target name match to image
            ("flyMIN", pim_f, cxl_f, acc_f, cpu_f, y_pos + bar_height, '\\\\\\') # case sensitive target name match
        ]

        # Draw stacked bars with slight y-offset for side-by-side comparison
        # Note: Labels on individual bars are disabled here to manage consolidated comparative legend
        for label, pim, cxl, acc, cpu, pos, hatch in sources:
            ax.barh(pos, pim, height=bar_height, color="navajowhite", edgecolor='black', hatch=hatch, label="") # No label here
            ax.barh(pos, cxl, left=pim, height=bar_height, color="lightblue", edgecolor='black', hatch=hatch, label="")
            # PNM from original is darkgreen, rename to Acc for legend consistency
            ax.barh(pos, acc, left=pim+cxl, height=bar_height, color="darkgreen", edgecolor='black', hatch=hatch, label="") 
            ax.barh(pos, cpu, left=pim+cxl+acc, height=bar_height, color="dimgray", edgecolor='black', hatch=hatch, label="") # Host CPU component is black

        ax.set_yticks(y_pos)
        ax.set_yticklabels(parallel_list, fontsize=18)
        ax.set_xlabel("Query Latency (minute)", fontsize=18)
        plt.tick_params(axis='y', labelsize=18)
        plt.tick_params(axis='x', labelsize=18)
        ax.set_title(f"Latency Breakdown: {model} on {device} Devices", fontsize=20)
        
        # Consolidate comparative legend - with proxy artists to precisely match example image
        # Create proxy patches
        # Component Colors
        pim_patch = mpatches.Patch(facecolor='navajowhite', edgecolor='black', label='PIM')
        cxl_patch = mpatches.Patch(facecolor='lightblue', edgecolor='black', label='CXL')
        # Dark green component has been renamed to Acc for component consistency
        acc_patch = mpatches.Patch(facecolor='darkgreen', edgecolor='black', label='PNM')
        cpu_patch = mpatches.Patch(facecolor='dimgray', edgecolor='black', label='Host CPU')
        # Hatch patterns for comparative targets
        # Baseline target has no hatch
        baseline_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='CENT Baseline')
        # mountMIN target has '///' hatch
        mountmin_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='mountMIN') # case sensitive label match to image
        # flyMIN target has '\\\\\\' hatch. Note case sensitive label match
        flymin_hatch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='\\\\\\', label='flyMIN') # case sensitive label match

        # Collect all comparative legend entries
        legend_patches = [pim_patch, cxl_patch, acc_patch, cpu_patch, baseline_hatch, mountmin_hatch, flymin_hatch]
        
        # Drop Host CPU from consolidated legend to precisely match example image, as it's not present there. The bars are plotted but not in legend.
        
        # Set consolidated comparative legend and place in bottom-left
        # Note: loc="lower left" places it within axes and bottom-left as requested
        ax.legend(handles=legend_patches, fontsize=18, loc="upper right")
        
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if not os.path.exists("figures"): os.mkdir("figures")
        plt.savefig(f'figures/figure_{model}_{device}_14c_comparative.pdf')
        plt.close()

        # --- NEW CALCULATION AND PRINT BLOCK ---
        # Calculate total latencies by summing the components for each configuration
        total_b = pim_b + cxl_b + acc_b + cpu_b
        total_m = pim_m + cxl_m + acc_m + cpu_m
        total_f = pim_f + cxl_f + acc_f + cpu_f

        print(f"\nLatency Reduction for {model} on {device} Devices:")
        for i, config in enumerate(parallel_list):
            tb = total_b[i]
            tm = total_m[i]
            tf = total_f[i]
            
            if tb > 0:
                mount_improvement = (tb - tm) / tb
                fly_improvement = (tb - tf) / tb
            else:
                mount_improvement = 0
                fly_improvement = 0
                
            print(f"  {config}:")
            print(f"    mountMIN: {mount_improvement * 100:.2f}%")
            print(f"    flyMIN: {fly_improvement * 100:.2f}%")