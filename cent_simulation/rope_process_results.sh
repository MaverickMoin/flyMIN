for phase in prefill decoding end2end
do
python3 run_sim_rope.py --model Llama2-7B --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv

python3 run_sim_rope.py --model Llama2-7B --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv

python3 run_sim_rope.py --model Llama2-7B --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv

python3 run_sim_rope.py --model Llama2-7B --model_parallel --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --model_parallel --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --model_parallel --process_throughputs --num_devices 8 --phase $phase --simulation_result_path ropesimulation_results.csv

python3 run_sim_rope.py --model Llama2-7B --model_parallel --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --model_parallel --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --model_parallel --process_throughputs --num_devices 20 --phase $phase --simulation_result_path ropesimulation_results.csv

python3 run_sim_rope.py --model Llama2-7B --model_parallel --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-13B --model_parallel --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv
python3 run_sim_rope.py --model Llama2-70B --model_parallel --process_throughputs --num_devices 32 --phase $phase --simulation_result_path ropesimulation_results.csv

done
