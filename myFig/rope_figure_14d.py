import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_baseline = pd.read_csv('cent_simulation/baseline_ropesimulation_results.csv')
df_mount = pd.read_csv('cent_simulation/mount_ropesimulation_results.csv')
df_fly = pd.read_csv('cent_simulation/fly_ropesimulation_results.csv')

prefill_size = 512
decoding_list = [128, 512, 1024, 3584]

# Define Model-Device Constraints
model_device_combinations = {
    'Llama2-7B': [8, 20, 32],
    'Llama2-13B': [20, 32],
    'Llama2-70B': [32]
}
transformer_blocks_dict = {'Llama2-7B': 32, 'Llama2-13B': 40, 'Llama2-70B': 80}

def get_14d_latencies(df_source, model, device, transformer_blocks):
    df = df_source[(df_source['Model'] == model) & (df_source['Device number'] == device) & 
                   (df_source['Pipeline parallelism'] == transformer_blocks) & (df_source['Tensor parallelism'] == 1)]
    if df.empty:
        return np.zeros(len(decoding_list)), np.zeros(len(decoding_list))

    df_prefill = df[(df['Sequence length'] <= prefill_size)]
    prefill_lat = ((df_prefill['PIM latency']+df_prefill['Acc latency']).mean() * transformer_blocks / 1000 / 60 * prefill_size ) if not df_prefill.empty else 0

    prefill_arr = np.full(len(decoding_list), prefill_lat)
    decoding_arr = []

    for d in decoding_list:
        df_dec = df[(df['Sequence length'] > prefill_size) & (df['Sequence length'] <= d + prefill_size)]
        dec_lat = ((df_dec['PIM latency']+df_dec['Acc latency']).mean() * transformer_blocks / 1000 / 60 * d ) if not df_dec.empty else 0
        decoding_arr.append(dec_lat)

    return prefill_arr, np.array(decoding_arr)

for model, device_numbers in model_device_combinations.items():
    for device in device_numbers:
        transformer_blocks = transformer_blocks_dict[model]
        
        pref_b, dec_b = get_14d_latencies(df_baseline, model, device, transformer_blocks)
        pref_m, dec_m = get_14d_latencies(df_mount, model, device, transformer_blocks)
        pref_f, dec_f = get_14d_latencies(df_fly, model, device, transformer_blocks)

        if sum(pref_b) == 0 and sum(pref_m) == 0 and sum(pref_f) == 0:
            continue # Skip if completely empty across sources
            
        # ---------------------------------------------------------
        # NEW SECTION: Calculate and Print Latency Reductions
        # ---------------------------------------------------------
        print(f"\n[{model} - {device} Devices (PP={transformer_blocks}, TP=1)] RoPE Latency Reductions:")
        for i, decode_len in enumerate(decoding_list):
            total_b = pref_b[i] + dec_b[i]
            total_m = pref_m[i] + dec_m[i]
            total_f = pref_f[i] + dec_f[i]
            
            if total_b > 0:
                red_mount = ((total_b - total_m) / total_b) * 100
                red_fly = ((total_b - total_f) / total_b) * 100
                
                print(f"  Decode Length {decode_len:4d}: "
                      f"mountMIN Reduction = {red_mount:6.2f}% | "
                      f"flyMIN Reduction = {red_fly:6.2f}%")
            else:
                print(f"  Decode Length {decode_len:4d}: Baseline is 0, cannot calculate reduction.")
        # ---------------------------------------------------------

        df_results = pd.DataFrame({
            'Decoding Length': decoding_list,
            'Baseline Prefill': pref_b, 'Baseline Decoding': dec_b,
            'Mount Prefill': pref_m, 'Mount Decoding': dec_m,
            'Fly Prefill': pref_f, 'Fly Decoding': dec_f
        })

        if not os.path.exists("figure_source_data"): os.mkdir("figure_source_data")
        df_results.to_csv(f'figure_source_data/figure_{model}_{device}_14d.csv', index=False)

        x_labels = [f"Decode(Out) \n {d}" for d in decoding_list]
        x = np.arange(len(x_labels))
        bar_width = 0.25 # Adjust for side-by-side vertical plotting

        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Using position offsets and hatching to visually distinguish sources
        sources = [
            ("CENT", pref_b, dec_b, x - bar_width, ''),
            ("MountMIN", pref_m, dec_m, x, '///'),
            ("Fly", pref_f, dec_f, x + bar_width, '\\\\\\')
        ]

        for label, prefill, decoding, pos, hatch in sources:
            # Prefill Bars
            ax.bar(pos, prefill, width=bar_width, color="mediumpurple", edgecolor='black', hatch=hatch, 
                   label=f"Prefill ({label})")
            # Decoding Bars
            ax.bar(pos, decoding, width=bar_width, bottom=prefill, color="dodgerblue", edgecolor='black', hatch=hatch, 
                   label=f"Decoding ({label})")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=16, rotation=0)
        ax.set_ylabel("Query Latency (minutes)", fontsize=16)
        ax.set_title(f"{model} - {device} Devices (Prefill-512)", fontsize=16)
        plt.tick_params(axis='y', labelsize=16)
        # Organize legend into 2 columns
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=12, loc="upper left", ncol=2)
        
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        if not os.path.exists("figures"): os.mkdir("figures")
        plt.savefig(f'figures/rope_figure_{model}_{device}_14d.pdf')
        plt.close()
