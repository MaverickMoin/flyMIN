# Pipeline Parallel
threads=$1
seqlen_gap=$2


# Model Parallel
python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap


