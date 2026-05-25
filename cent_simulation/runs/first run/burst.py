import pandas as pd
import json

# 1. Load the JSON as a standard dictionary
file_path = 'logs/Llama2-7B_all_traces_burst_stats.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# 2. Use json_normalize to flatten the nested "flags" and other keys.
# We pass the dictionary values and use the keys as the 'trace' column.
df = pd.json_normalize(data)

# 3. Add the trace file path as the first column (the original keys)
cols = ['traces'] + [c for c in df.columns if c != 'traces']
df = df[cols]

# Optional: Print the first few rows to verify
print(df.head())

# Optional: Export to CSV
df.to_csv('burst_stats_analysis.csv', index=False)

