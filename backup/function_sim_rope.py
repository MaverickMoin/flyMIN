import torch
from utils import get_args, compare
from TBLlama import TransformerBlockLlama
import json

if __name__ == "__main__":

    args = get_args()
    if args.filename:
        dic_model = torch.load(args.filename)
    else:
        head_dim = 128
        dim = head_dim * args.n_heads
        ffn_dim = args.ffn_dim
        TP_param = 8 if args.GPT3_175B_TP_8 else 1
        n_heads = args.n_heads // TP_param
        n_kv_heads = args.n_kv_heads if args.Llama_GQA else n_heads
        dic_model = {
            "TP_param": torch.tensor(TP_param),
            "dim": torch.tensor(dim),
            "n_heads": torch.tensor(n_heads),
            "x": torch.zeros((1, 1, dim)),
            "SANorm": torch.zeros(dim),
            "FFNNorm": torch.zeros(dim),
            "sa": torch.zeros((1, 1, dim)),
            "h": torch.zeros((1, 1, dim)),
            "out": torch.zeros((1, 1, dim)),
            "wq": torch.zeros((dim // TP_param, dim)),
            "wk": torch.zeros((head_dim * n_kv_heads), dim),
            "wv": torch.zeros((head_dim * n_kv_heads), dim),
            "xq": torch.zeros((1, 1, dim)),
            "xk": torch.zeros((1, 1, head_dim * n_heads)),
            "xv": torch.zeros((1, 1, head_dim * n_heads)),
            "start_pos": torch.tensor(args.seqlen - 1),
            "cache_k": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
            "cache_v": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
            "scores": torch.zeros((1, n_heads, 1, args.seqlen)),
            "output": torch.zeros((1, 1, dim)),
            "wo": torch.zeros((dim // TP_param, dim)),
            "w1": torch.zeros((ffn_dim // TP_param, dim)),
            "w3": torch.zeros((ffn_dim // TP_param, dim)),
            "w2": torch.zeros((dim // TP_param, ffn_dim)),
            "ffn": torch.zeros((1, 1, dim))
        }
    if args.Llama_GQA:
        dic_model["n_kv_heads"] = torch.tensor(n_kv_heads)

    print("\n generate trace in function_sim, model name={}\n".format(args.model))
    
    TB = TransformerBlockLlama(dic_model, args)
    # print("Variable\t Dimension\t\t\t Rows required\n")
    TB.memory_mapping()

    if args.only_trace:
        TB.trace_only()
        TB.finish()
        TB.file.close()
        
        # MAINUDDIN
        import json
        
        # Extract the burst stats directly from the TB object
        burst_stats = {
            "rd_q_burst": TB.rd_q_burst,
            "store_for_rd_q_burst": TB.store_for_rd_q_burst,
            "rd_k_burst": TB.rd_k_burst,
            "store_for_rd_k_burst": TB.store_for_rd_k_burst,
            "EWMUL_Q": TB.ewmul_q_burst,
            "EWMUL_K": TB.ewmul_k_burst,
            "inv_q_burst": TB.inv_q_burst,
            "store_for_inv_q_burst": TB.store_for_inv_q_burst,
            "inv_k_burst": TB.inv_k_burst,
            "store_for_inv_k_burst": TB.store_for_inv_k_burst
        }
           
        # Save to a JSON file uniquely named after this specific trace run
        # (Assuming your get_args() parser maps '--trace-file' to args.trace_file)
        if hasattr(args, 'trace_file') and args.trace_file:
            stats_file_path = f"{args.trace_file}.json"
            with open(stats_file_path, "w") as f:
                json.dump(burst_stats, f, indent=4)
        # MAINUDDIN
        